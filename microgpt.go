// The most atomic way to train and run inference for a GPT in pure, dependency-free Go.
// This file is the complete algorithm.
// Everything else is just efficiency.
//
// Ported from @karpathy's microgpt.py — reimagined with tape-based autograd.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

// Internal constants (rarely changed)
const (
	initStd      = 0.08
	beta1        = 0.85
	beta2        = 0.99
	epsAdam      = 1e-8
	tapeInitCap  = 50_000
	gradClipNorm = 1.0  // max global gradient L2 norm
	warmupFrac   = 0.05 // fraction of training steps for LR warmup (0→lr)
)

// Config holds model hyperparameters, eliminating global mutable state.
type Config struct {
	NLayer    int
	NEmbd     int
	BlockSize int
	NHead     int
	HeadDim   int
}

// ---------------------------------------------------------------------------
// Autograd: tape-based (Wengert list)
// ---------------------------------------------------------------------------

// Idx is a typed index into the tape, preventing accidental use as a plain int.
type Idx int32

type opKind uint8

const (
	opLeaf opKind = iota
	opAdd
	opMul
	opPow
	opLog
	opExp
	opReLU
	opNeg
	opSub
	opDiv
	opDot
)

type node struct {
	data, grad float64
	op         opKind
	a, b       Idx // For opDot: a = index into dotMeta (not a node index), b unused.
}

// Tape is a linear list of operations. Nodes are in topological order by
// construction: every operand index is strictly less than the node that
// references it. Backward is therefore a simple reverse scan.
type Tape struct {
	nodes   []node
	powExps []float64 // dense storage for opPow exponents (indexed via node.b)
	dotBuf  []Idx     // packed operand indices for opDot: [a0..aN-1, b0..bN-1, ...]
	dotMeta []int32   // pairs of (offset_into_dotBuf, length) for each opDot
}

func NewTape(cap int) *Tape {
	return &Tape{
		nodes:   make([]node, 0, cap),
		powExps: make([]float64, 0, cap/16),
		dotBuf:  make([]Idx, 0, cap),
		dotMeta: make([]int32, 0, cap/8),
	}
}

// Reset clears the tape for reuse, keeping the backing arrays.
func (t *Tape) Reset() {
	t.nodes = t.nodes[:0]
	t.powExps = t.powExps[:0]
	t.dotBuf = t.dotBuf[:0]
	t.dotMeta = t.dotMeta[:0]
}

func (t *Tape) push(n node) Idx {
	idx := Idx(len(t.nodes))
	t.nodes = append(t.nodes, n)
	return idx
}

// Primitives

func (t *Tape) Leaf(data float64) Idx {
	return t.push(node{data: data, op: opLeaf})
}

func (t *Tape) Add(a, b Idx) Idx {
	return t.push(node{data: t.nodes[a].data + t.nodes[b].data, op: opAdd, a: a, b: b})
}

func (t *Tape) Mul(a, b Idx) Idx {
	return t.push(node{data: t.nodes[a].data * t.nodes[b].data, op: opMul, a: a, b: b})
}

func (t *Tape) Pow(a Idx, exp float64) Idx {
	expIdx := Idx(len(t.powExps))
	t.powExps = append(t.powExps, exp)
	return t.push(node{data: math.Pow(t.nodes[a].data, exp), op: opPow, a: a, b: expIdx})
}

func (t *Tape) Log(a Idx) Idx {
	return t.push(node{data: math.Log(t.nodes[a].data), op: opLog, a: a})
}

func (t *Tape) Exp(a Idx) Idx {
	return t.push(node{data: math.Exp(t.nodes[a].data), op: opExp, a: a})
}

func (t *Tape) ReLU(a Idx) Idx {
	d := t.nodes[a].data
	if d < 0 {
		d = 0
	}
	return t.push(node{data: d, op: opReLU, a: a})
}

// Compound ops (now native primitives for efficiency)

func (t *Tape) Neg(a Idx) Idx {
	return t.push(node{data: -t.nodes[a].data, op: opNeg, a: a})
}

func (t *Tape) Sub(a, b Idx) Idx {
	return t.push(node{data: t.nodes[a].data - t.nodes[b].data, op: opSub, a: a, b: b})
}

func (t *Tape) Div(a, b Idx) Idx {
	return t.push(node{data: t.nodes[a].data / t.nodes[b].data, op: opDiv, a: a, b: b})
}

// Dot computes the dot product of two equal-length vectors as a single fused node.
// This replaces 2N-1 scalar nodes (N Mul + N-1 Add) with 1 node.
func (t *Tape) Dot(as, bs []Idx) Idx {
	if len(as) != len(bs) {
		panic("Dot: mismatched operand lengths")
	}
	sum := 0.0
	for i := range as {
		sum += t.nodes[as[i]].data * t.nodes[bs[i]].data
	}
	off := int32(len(t.dotBuf))
	t.dotBuf = append(t.dotBuf, as...)
	t.dotBuf = append(t.dotBuf, bs...)
	metaIdx := Idx(len(t.dotMeta) / 2)
	t.dotMeta = append(t.dotMeta, off, int32(len(as)))
	return t.push(node{data: sum, op: opDot, a: metaIdx})
}

func (t *Tape) Sum(xs []Idx) Idx {
	switch len(xs) {
	case 0:
		return t.Leaf(0)
	case 1:
		return xs[0]
	}
	s := xs[0]
	for _, x := range xs[1:] {
		s = t.Add(s, x)
	}
	return s
}

// Accessors

func (t *Tape) Val(idx Idx) float64  { return t.nodes[idx].data }
func (t *Tape) Grad(idx Idx) float64 { return t.nodes[idx].grad }

// Backward propagates gradients from root to all leaves.
func (t *Tape) Backward(root Idx) {
	t.nodes[root].grad = 1
	for i := int(root); i >= 0; i-- {
		n := &t.nodes[i]
		if n.grad == 0 {
			continue
		}
		switch n.op {
		case opLeaf:
			// no children
		case opAdd:
			t.nodes[n.a].grad += n.grad
			t.nodes[n.b].grad += n.grad
		case opMul:
			t.nodes[n.a].grad += t.nodes[n.b].data * n.grad
			t.nodes[n.b].grad += t.nodes[n.a].data * n.grad
		case opPow:
			exp := t.powExps[n.b]
			t.nodes[n.a].grad += exp * math.Pow(t.nodes[n.a].data, exp-1) * n.grad
		case opLog:
			t.nodes[n.a].grad += n.grad / t.nodes[n.a].data
		case opExp:
			t.nodes[n.a].grad += n.data * n.grad
		case opReLU:
			if t.nodes[n.a].data > 0 {
				t.nodes[n.a].grad += n.grad
			}
		case opNeg:
			t.nodes[n.a].grad += -n.grad
		case opSub:
			t.nodes[n.a].grad += n.grad
			t.nodes[n.b].grad += -n.grad
		case opDiv:
			bData := t.nodes[n.b].data
			t.nodes[n.a].grad += n.grad / bData
			t.nodes[n.b].grad += -t.nodes[n.a].data * n.grad / (bData * bData)
		case opDot:
			off := t.dotMeta[n.a*2]
			length := t.dotMeta[n.a*2+1]
			as := t.dotBuf[off : off+length]
			bs := t.dotBuf[off+length : off+2*length]
			for j := range length {
				t.nodes[as[j]].grad += t.nodes[bs[j]].data * n.grad
				t.nodes[bs[j]].grad += t.nodes[as[j]].data * n.grad
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Tokenizer: char-level, with a BOS special token
// ---------------------------------------------------------------------------

type Tokenizer struct {
	chars     []rune
	charToID  map[rune]int
	bos       int
	vocabSize int
}

func NewTokenizer(docs []string) *Tokenizer {
	seen := make(map[rune]bool)
	for _, doc := range docs {
		for _, ch := range doc { // iterate over Unicode code points
			seen[ch] = true
		}
	}
	chars := make([]rune, 0, len(seen))
	for ch := range seen {
		chars = append(chars, ch)
	}
	slices.Sort(chars)

	charToID := make(map[rune]int, len(chars))
	for i, ch := range chars {
		charToID[ch] = i
	}
	return &Tokenizer{
		chars:     chars,
		charToID:  charToID,
		bos:       len(chars),
		vocabSize: len(chars) + 1,
	}
}

// Encode returns [BOS, char_ids..., BOS].
func (tk *Tokenizer) Encode(doc string) []int {
	tokens := make([]int, 0, len([]rune(doc))+2)
	tokens = append(tokens, tk.bos)
	for _, ch := range doc { // iterate over Unicode code points
		tokens = append(tokens, tk.charToID[ch])
	}
	return append(tokens, tk.bos)
}

// Decode maps a token id back to a rune (comma-ok pattern).
func (tk *Tokenizer) Decode(id int) (rune, bool) {
	if id < 0 || id >= len(tk.chars) {
		return 0, false
	}
	return tk.chars[id], true
}

// ---------------------------------------------------------------------------
// Model types & parameter loading
// ---------------------------------------------------------------------------

// Params holds the flat weight vector and Adam optimizer buffers.
type Params struct{ data, m, v []float64 }

// Layer holds tape indices for one transformer layer's weight matrices.
type Layer struct {
	attnWQ, attnWK, attnWV, attnWO [][]Idx
	mlpFC1, mlpFC2                 [][]Idx
}

// Model holds tape indices for all weight matrices, rebuilt each step.
type Model struct {
	wte, wpe, lmHead [][]Idx
	layers           []Layer
}

type kvCache struct {
	keys   [][][]Idx
	values [][][]Idx
}

func NewParams(cfg *Config, vocabSize int, rng *rand.Rand) *Params {
	n := vocabSize*cfg.NEmbd + // wte
		cfg.BlockSize*cfg.NEmbd + // wpe
		vocabSize*cfg.NEmbd + // lmHead
		cfg.NLayer*(cfg.NEmbd*cfg.NEmbd+ // attnWQ
			cfg.NEmbd*cfg.NEmbd+ // attnWK
			cfg.NEmbd*cfg.NEmbd+ // attnWV
			cfg.NEmbd*cfg.NEmbd+ // attnWO
			4*cfg.NEmbd*cfg.NEmbd+ // mlpFC1
			cfg.NEmbd*4*cfg.NEmbd) // mlpFC2

	data := make([]float64, n)
	for i := range data {
		data[i] = rng.NormFloat64() * initStd
	}
	return &Params{data: data, m: make([]float64, n), v: make([]float64, n)}
}

// loadModel creates tape leaves for all params and carves them into matrices.
func loadModel(t *Tape, p *Params, cfg *Config, vocabSize int) (Model, []Idx) {
	allIdx := make([]Idx, len(p.data))
	for i, d := range p.data {
		allIdx[i] = t.Leaf(d)
	}

	off := 0
	matrix := func(rows, cols int) [][]Idx {
		m := make([][]Idx, rows)
		for r := range rows {
			m[r] = allIdx[off : off+cols : off+cols]
			off += cols
		}
		return m
	}

	var mdl Model
	mdl.wte = matrix(vocabSize, cfg.NEmbd)
	mdl.wpe = matrix(cfg.BlockSize, cfg.NEmbd)
	mdl.lmHead = matrix(vocabSize, cfg.NEmbd)
	mdl.layers = make([]Layer, cfg.NLayer)
	for i := range cfg.NLayer {
		mdl.layers[i] = Layer{
			attnWQ: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWK: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWV: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWO: matrix(cfg.NEmbd, cfg.NEmbd),
			mlpFC1: matrix(4*cfg.NEmbd, cfg.NEmbd),
			mlpFC2: matrix(cfg.NEmbd, 4*cfg.NEmbd),
		}
	}
	return mdl, allIdx
}

func newKVCache(nLayer int) *kvCache {
	return &kvCache{
		keys:   make([][][]Idx, nLayer),
		values: make([][][]Idx, nLayer),
	}
}

// ---------------------------------------------------------------------------
// Neural network ops (composite, not intrinsic to tape)
// ---------------------------------------------------------------------------

func linear(t *Tape, x []Idx, w [][]Idx) []Idx {
	out := make([]Idx, len(w))
	for i, row := range w {
		out[i] = t.Dot(row, x)
	}
	return out
}

func linearInto(t *Tape, x []Idx, w [][]Idx, out []Idx) {
	for i, row := range w {
		out[i] = t.Dot(row, x)
	}
}

func softmax(t *Tape, logits []Idx) []Idx {
	if len(logits) == 0 {
		panic("softmax: empty logits")
	}
	maxVal := t.Val(logits[0])
	for _, l := range logits[1:] {
		if v := t.Val(l); v > maxVal {
			maxVal = v
		}
	}
	maxC := t.Leaf(maxVal)

	exps := make([]Idx, len(logits))
	for i, l := range logits {
		exps[i] = t.Exp(t.Sub(l, maxC))
	}
	total := t.Sum(exps)

	probs := make([]Idx, len(logits))
	for i, e := range exps {
		probs[i] = t.Div(e, total)
	}
	return probs
}

func rmsnorm(t *Tape, x []Idx) []Idx {
	n := len(x)
	sq := make([]Idx, n)
	for i, xi := range x {
		sq[i] = t.Mul(xi, xi)
	}
	ms := t.Mul(t.Sum(sq), t.Leaf(1.0/float64(n)))
	scale := t.Pow(t.Add(ms, t.Leaf(1e-5)), -0.5)

	out := make([]Idx, n)
	for i, xi := range x {
		out[i] = t.Mul(xi, scale)
	}
	return out
}

func rmsnormInto(t *Tape, x, sq, out []Idx) {
	n := len(x)
	for i, xi := range x {
		sq[i] = t.Mul(xi, xi)
	}
	ms := t.Mul(t.Sum(sq[:n]), t.Leaf(1.0/float64(n)))
	scale := t.Pow(t.Add(ms, t.Leaf(1e-5)), -0.5)
	for i, xi := range x {
		out[i] = t.Mul(xi, scale)
	}
}

// ---------------------------------------------------------------------------
// Training workspace — pre-allocated buffers reused across positions
// ---------------------------------------------------------------------------

type fwdWorkspace struct {
	x, xRes    []Idx // [nEmbd]
	q, k, v    []Idx // [nEmbd]
	xAttn      []Idx // [nEmbd]
	hidden     []Idx // [4*nEmbd]
	normSq     []Idx // [nEmbd] — rmsnorm scratch
	attnLogits []Idx // [blockSize] — attention scores per head
	prods      []Idx // [blockSize] — weighted sum products per head
}

func newFwdWorkspace(cfg *Config) *fwdWorkspace {
	return &fwdWorkspace{
		x:          make([]Idx, cfg.NEmbd),
		xRes:       make([]Idx, cfg.NEmbd),
		q:          make([]Idx, cfg.NEmbd),
		k:          make([]Idx, cfg.NEmbd),
		v:          make([]Idx, cfg.NEmbd),
		xAttn:      make([]Idx, cfg.NEmbd),
		hidden:     make([]Idx, 4*cfg.NEmbd),
		normSq:     make([]Idx, cfg.NEmbd),
		attnLogits: make([]Idx, cfg.BlockSize),
		prods:      make([]Idx, cfg.BlockSize),
	}
}

// ---------------------------------------------------------------------------
// GPT forward pass (single token, KV-cache style)
// ---------------------------------------------------------------------------

func gptForward(t *Tape, m *Model, cfg *Config, tokenID, posID int, kv *kvCache, ws *fwdWorkspace) []Idx {
	// Token + position embedding
	tokEmb := m.wte[tokenID]
	posEmb := m.wpe[posID]
	for i := range cfg.NEmbd {
		ws.x[i] = t.Add(tokEmb[i], posEmb[i])
	}
	rmsnormInto(t, ws.x, ws.normSq, ws.x)

	scale := t.Leaf(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	for li := range cfg.NLayer {
		layer := &m.layers[li]

		// --- Multi-head attention ---
		copy(ws.xRes, ws.x)
		rmsnormInto(t, ws.x, ws.normSq, ws.x)
		linearInto(t, ws.x, layer.attnWQ, ws.q)
		linearInto(t, ws.x, layer.attnWK, ws.k)
		linearInto(t, ws.x, layer.attnWV, ws.v)

		// KV cache entries must persist across positions — allocate per position
		kEntry := make([]Idx, cfg.NEmbd)
		copy(kEntry, ws.k)
		vEntry := make([]Idx, cfg.NEmbd)
		copy(vEntry, ws.v)
		kv.keys[li] = append(kv.keys[li], kEntry)
		kv.values[li] = append(kv.values[li], vEntry)

		for h := range cfg.NHead {
			hs := h * cfg.HeadDim
			qH := ws.q[hs : hs+cfg.HeadDim]
			nPos := len(kv.keys[li])

			for p := range nPos {
				kH := kv.keys[li][p][hs : hs+cfg.HeadDim]
				ws.attnLogits[p] = t.Mul(t.Dot(qH, kH), scale)
			}
			attnWeights := softmax(t, ws.attnLogits[:nPos])

			for j := range cfg.HeadDim {
				for p := range nPos {
					ws.prods[p] = t.Mul(attnWeights[p], kv.values[li][p][hs+j])
				}
				ws.xAttn[hs+j] = t.Sum(ws.prods[:nPos])
			}
		}
		linearInto(t, ws.xAttn, layer.attnWO, ws.x)
		for i := range cfg.NEmbd {
			ws.x[i] = t.Add(ws.x[i], ws.xRes[i])
		}

		// --- MLP ---
		copy(ws.xRes, ws.x)
		rmsnormInto(t, ws.x, ws.normSq, ws.x)
		linearInto(t, ws.x, layer.mlpFC1, ws.hidden)
		for i := range ws.hidden {
			ws.hidden[i] = t.ReLU(ws.hidden[i])
		}
		linearInto(t, ws.hidden, layer.mlpFC2, ws.x)
		for i := range cfg.NEmbd {
			ws.x[i] = t.Add(ws.x[i], ws.xRes[i])
		}
	}

	return linear(t, ws.x, m.lmHead)
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

// computeLR returns the learning rate for the given step using linear warmup + linear decay.
func computeLR(baseLR float64, relStep, totalSteps int) float64 {
	warmup := float64(totalSteps) * warmupFrac
	rel := float64(relStep)
	if warmup > 0 && rel < warmup {
		return baseLR * rel / warmup
	}
	return baseLR * (float64(totalSteps) - rel) / (float64(totalSteps) - warmup)
}

func trainStep(params *Params, cfg *Config, vocabSize int, tokens []int, step int, lrT float64, tape *Tape) float64 {
	var t *Tape
	if tape != nil {
		t = tape
		t.Reset()
	} else {
		t = NewTape(tapeInitCap)
	}
	m, allIdx := loadModel(t, params, cfg, vocabSize)
	kv := newKVCache(cfg.NLayer)
	ws := newFwdWorkspace(cfg)

	n := min(cfg.BlockSize, len(tokens)-1)

	losses := make([]Idx, n)
	for pos := range n {
		logits := gptForward(t, &m, cfg, tokens[pos], pos, kv, ws)
		probs := softmax(t, logits)
		losses[pos] = t.Neg(t.Log(probs[tokens[pos+1]]))
	}

	loss := t.Mul(t.Leaf(1.0/float64(n)), t.Sum(losses))
	t.Backward(loss)

	// Global gradient norm clipping: scale all gradients so ||g|| <= gradClipNorm.
	// If the norm is NaN/Inf (forward pass exploded), skip the Adam update entirely
	// so params stay at their last valid values and the caller can save a clean checkpoint.
	var gradNormSq float64
	for _, idx := range allIdx {
		g := t.Grad(idx)
		gradNormSq += g * g
	}
	if math.IsNaN(gradNormSq) || math.IsInf(gradNormSq, 0) {
		return t.Val(loss)
	}
	clipScale := 1.0
	if gradNorm := math.Sqrt(gradNormSq); gradNorm > gradClipNorm {
		clipScale = gradClipNorm / gradNorm
	}

	s := float64(step + 1)
	bc1 := 1 - math.Pow(beta1, s)
	bc2 := 1 - math.Pow(beta2, s)
	for i, idx := range allIdx {
		g := t.Grad(idx) * clipScale
		params.m[i] = beta1*params.m[i] + (1-beta1)*g
		params.v[i] = beta2*params.v[i] + (1-beta2)*g*g
		mHat := params.m[i] / bc1
		vHat := params.v[i] / bc2
		params.data[i] -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
	}

	return t.Val(loss)
}

// ---------------------------------------------------------------------------
// Batched training (data parallelism)
// ---------------------------------------------------------------------------

// trainWorker holds per-document state for parallel training.
type trainWorker struct {
	tape   *Tape
	ws     *fwdWorkspace
	allIdx []Idx
	loss   Idx
}

func newTrainWorkers(n int, cfg *Config) []trainWorker {
	workers := make([]trainWorker, n)
	for i := range workers {
		workers[i] = trainWorker{
			tape: NewTape(tapeInitCap),
			ws:   newFwdWorkspace(cfg),
		}
	}
	return workers
}

// trainStepBatch processes multiple documents in parallel and performs one Adam update
// on the averaged gradients. When len(tokensBatch)==1, it is mathematically equivalent
// to trainStep (same gradients, same update).
func trainStepBatch(params *Params, cfg *Config, vocabSize int, tokensBatch [][]int, step int, lrT float64, workers []trainWorker) float64 {
	batchSize := len(tokensBatch)
	invBatch := 1.0 / float64(batchSize)

	// Phase 1: parallel forward + backward (each worker has its own tape).
	var wg sync.WaitGroup
	for wi := range batchSize {
		wg.Add(1)
		go func(wi int) {
			defer wg.Done()
			w := &workers[wi]
			w.tape.Reset()
			m, allIdx := loadModel(w.tape, params, cfg, vocabSize)
			w.allIdx = allIdx
			kv := newKVCache(cfg.NLayer)

			tokens := tokensBatch[wi]
			n := min(cfg.BlockSize, len(tokens)-1)
			losses := make([]Idx, n)
			for pos := range n {
				logits := gptForward(w.tape, &m, cfg, tokens[pos], pos, kv, w.ws)
				probs := softmax(w.tape, logits)
				losses[pos] = w.tape.Neg(w.tape.Log(probs[tokens[pos+1]]))
			}
			w.loss = w.tape.Mul(w.tape.Leaf(1.0/float64(n)), w.tape.Sum(losses))
			w.tape.Backward(w.loss)
		}(wi)
	}
	wg.Wait()

	// Average loss for reporting.
	var avgLoss float64
	for wi := range batchSize {
		avgLoss += workers[wi].tape.Val(workers[wi].loss)
	}
	avgLoss *= invBatch

	nParams := len(params.data)

	// Phase 2: parallel gradient aggregation + norm computation.
	// Split parameter range across OS threads to parallelise the reduction.
	// For small param counts, sequential is faster than goroutine overhead.
	nChunks := 1
	if nParams >= 4096 {
		nChunks = runtime.GOMAXPROCS(0)
		if nChunks > nParams/1024 {
			nChunks = nParams / 1024
		}
	}
	chunkSize := (nParams + nChunks - 1) / nChunks

	// Temporary buffer for averaged gradients (avoids re-reading from tapes in Adam).
	avgGrad := make([]float64, nParams)
	partialNormSq := make([]float64, nChunks)

	var wg2 sync.WaitGroup
	for c := range nChunks {
		wg2.Add(1)
		go func(c int) {
			defer wg2.Done()
			start := c * chunkSize
			end := start + chunkSize
			if end > nParams {
				end = nParams
			}
			var localNormSq float64
			for j := start; j < end; j++ {
				var sum float64
				for wi := range batchSize {
					sum += workers[wi].tape.Grad(workers[wi].allIdx[j])
				}
				g := sum * invBatch
				avgGrad[j] = g
				localNormSq += g * g
			}
			partialNormSq[c] = localNormSq
		}(c)
	}
	wg2.Wait()

	var gradNormSq float64
	for _, ps := range partialNormSq {
		gradNormSq += ps
	}
	if math.IsNaN(gradNormSq) || math.IsInf(gradNormSq, 0) {
		return avgLoss
	}

	clipScale := 1.0
	if gradNorm := math.Sqrt(gradNormSq); gradNorm > gradClipNorm {
		clipScale = gradClipNorm / gradNorm
	}

	// Phase 3: parallel Adam update.
	s := float64(step + 1)
	bc1 := 1 - math.Pow(beta1, s)
	bc2 := 1 - math.Pow(beta2, s)

	var wg3 sync.WaitGroup
	for c := range nChunks {
		wg3.Add(1)
		go func(c int) {
			defer wg3.Done()
			start := c * chunkSize
			end := start + chunkSize
			if end > nParams {
				end = nParams
			}
			for i := start; i < end; i++ {
				g := avgGrad[i] * clipScale
				params.m[i] = beta1*params.m[i] + (1-beta1)*g
				params.v[i] = beta2*params.v[i] + (1-beta2)*g*g
				mHat := params.m[i] / bc1
				vHat := params.v[i] / bc2
				params.data[i] -= lrT * mHat / (math.Sqrt(vHat) + epsAdam)
			}
		}(c)
	}
	wg3.Wait()

	return avgLoss
}

// ---------------------------------------------------------------------------
// Inference (no autograd needed)
// ---------------------------------------------------------------------------

func softmaxF64Into(logits, probs []float64) {
	if len(logits) == 0 {
		panic("softmaxF64Into: empty logits")
	}
	maxVal := logits[0]
	for _, l := range logits[1:] {
		if l > maxVal {
			maxVal = l
		}
	}
	sum := 0.0
	for i, l := range logits {
		probs[i] = math.Exp(l - maxVal)
		sum += probs[i]
	}
	for i := range logits {
		probs[i] /= sum
	}
}

func weightedSample(probs []float64, rng *rand.Rand) int {
	r := rng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}

// ---------------------------------------------------------------------------
// Tape-free inference types and ops (pure float64, no autograd overhead)
// ---------------------------------------------------------------------------

type inferLayer struct {
	attnWQ, attnWK, attnWV, attnWO [][]float64
	mlpFC1, mlpFC2                 [][]float64
}

type inferModel struct {
	wte, wpe, lmHead [][]float64
	layers           []inferLayer
}

type inferKV struct {
	keys, values [][][]float64
}

func loadInferModel(p *Params, cfg *Config, vocabSize int) inferModel {
	off := 0
	matrix := func(rows, cols int) [][]float64 {
		m := make([][]float64, rows)
		for r := range rows {
			m[r] = p.data[off : off+cols : off+cols]
			off += cols
		}
		return m
	}

	var mdl inferModel
	mdl.wte = matrix(vocabSize, cfg.NEmbd)
	mdl.wpe = matrix(cfg.BlockSize, cfg.NEmbd)
	mdl.lmHead = matrix(vocabSize, cfg.NEmbd)
	mdl.layers = make([]inferLayer, cfg.NLayer)
	for i := range cfg.NLayer {
		mdl.layers[i] = inferLayer{
			attnWQ: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWK: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWV: matrix(cfg.NEmbd, cfg.NEmbd),
			attnWO: matrix(cfg.NEmbd, cfg.NEmbd),
			mlpFC1: matrix(4*cfg.NEmbd, cfg.NEmbd),
			mlpFC2: matrix(cfg.NEmbd, 4*cfg.NEmbd),
		}
	}
	return mdl
}

func dotF64(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func linearF64Into(x []float64, w [][]float64, out []float64) {
	for i, row := range w {
		out[i] = dotF64(row, x)
	}
}

func rmsnormF64Into(x, out []float64) {
	n := len(x)
	ms := 0.0
	for _, xi := range x {
		ms += xi * xi
	}
	ms /= float64(n)
	scale := 1.0 / math.Sqrt(ms+1e-5)
	for i, xi := range x {
		out[i] = xi * scale
	}
}

// ---------------------------------------------------------------------------
// Inference workspace — pre-allocated buffers reused across samples
// ---------------------------------------------------------------------------

type inferWorkspace struct {
	x, xRes    []float64 // [nEmbd]
	q, k, v    []float64 // [nEmbd]
	xAttn      []float64 // [nEmbd]
	hidden     []float64 // [4*nEmbd]
	attnLogits []float64 // [blockSize]
	attnW      []float64 // [blockSize] — softmax output
	logits     []float64 // [vocabSize]
	probs      []float64 // [vocabSize] — softmax output for sampling
}

func newInferWorkspace(cfg *Config, vocabSize int) *inferWorkspace {
	return &inferWorkspace{
		x:          make([]float64, cfg.NEmbd),
		xRes:       make([]float64, cfg.NEmbd),
		q:          make([]float64, cfg.NEmbd),
		k:          make([]float64, cfg.NEmbd),
		v:          make([]float64, cfg.NEmbd),
		xAttn:      make([]float64, cfg.NEmbd),
		hidden:     make([]float64, 4*cfg.NEmbd),
		attnLogits: make([]float64, cfg.BlockSize),
		attnW:      make([]float64, cfg.BlockSize),
		logits:     make([]float64, vocabSize),
		probs:      make([]float64, vocabSize),
	}
}

func gptForwardF64(m *inferModel, cfg *Config, tokenID, posID int, kv *inferKV, ws *inferWorkspace) {
	tokEmb := m.wte[tokenID]
	posEmb := m.wpe[posID]
	for i := range cfg.NEmbd {
		ws.x[i] = tokEmb[i] + posEmb[i]
	}
	rmsnormF64Into(ws.x, ws.x)

	scale := 1.0 / math.Sqrt(float64(cfg.HeadDim))

	for li := range cfg.NLayer {
		layer := &m.layers[li]

		copy(ws.xRes, ws.x)
		rmsnormF64Into(ws.x, ws.x)
		linearF64Into(ws.x, layer.attnWQ, ws.q)
		linearF64Into(ws.x, layer.attnWK, ws.k)
		linearF64Into(ws.x, layer.attnWV, ws.v)

		// KV cache entries must persist
		kEntry := make([]float64, cfg.NEmbd)
		copy(kEntry, ws.k)
		vEntry := make([]float64, cfg.NEmbd)
		copy(vEntry, ws.v)
		kv.keys[li] = append(kv.keys[li], kEntry)
		kv.values[li] = append(kv.values[li], vEntry)

		for h := range cfg.NHead {
			hs := h * cfg.HeadDim
			qH := ws.q[hs : hs+cfg.HeadDim]
			nPos := len(kv.keys[li])

			for p := range nPos {
				kH := kv.keys[li][p][hs : hs+cfg.HeadDim]
				ws.attnLogits[p] = dotF64(qH, kH) * scale
			}
			softmaxF64Into(ws.attnLogits[:nPos], ws.attnW[:nPos])

			for j := range cfg.HeadDim {
				s := 0.0
				for p := range nPos {
					s += ws.attnW[p] * kv.values[li][p][hs+j]
				}
				ws.xAttn[hs+j] = s
			}
		}
		linearF64Into(ws.xAttn, layer.attnWO, ws.x)
		for i := range cfg.NEmbd {
			ws.x[i] += ws.xRes[i]
		}

		copy(ws.xRes, ws.x)
		rmsnormF64Into(ws.x, ws.x)
		linearF64Into(ws.x, layer.mlpFC1, ws.hidden)
		for i := range ws.hidden {
			if ws.hidden[i] < 0 {
				ws.hidden[i] = 0
			}
		}
		linearF64Into(ws.hidden, layer.mlpFC2, ws.x)
		for i := range cfg.NEmbd {
			ws.x[i] += ws.xRes[i]
		}
	}

	linearF64Into(ws.x, m.lmHead, ws.logits)
}

func generate(params *Params, tok *Tokenizer, cfg *Config, temperature float64, rng *rand.Rand) string {
	m := loadInferModel(params, cfg, tok.vocabSize)
	kv := &inferKV{
		keys:   make([][][]float64, cfg.NLayer),
		values: make([][][]float64, cfg.NLayer),
	}
	ws := newInferWorkspace(cfg, tok.vocabSize)

	tokenID := tok.bos
	var sb strings.Builder
	for pos := range cfg.BlockSize {
		gptForwardF64(&m, cfg, tokenID, pos, kv, ws)

		for i := range ws.logits {
			ws.logits[i] /= temperature
		}
		softmaxF64Into(ws.logits, ws.probs)
		tokenID = weightedSample(ws.probs, rng)
		if tokenID == tok.bos {
			break
		}
		if ch, ok := tok.Decode(tokenID); ok {
			sb.WriteRune(ch)
		}
	}
	return sb.String()
}

// ---------------------------------------------------------------------------
// Checkpoint save / load
// ---------------------------------------------------------------------------

var checkpointMagic = [8]byte{'M', 'G', 'P', 'T'}

// Version 2: tokenizer chars stored as int32(numRunes) + int32(numBytes) + utf8 bytes.
// This supports arbitrary Unicode (multi-byte scripts like Cyrillic, CJK, etc.).
const checkpointVersion uint32 = 2

func saveCheckpoint(path string, params *Params, tok *Tokenizer, cfg *Config, step int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	order := binary.LittleEndian

	// Header
	if _, err := w.Write(checkpointMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, order, checkpointVersion); err != nil {
		return err
	}

	// Hyperparameters
	for _, v := range []int32{int32(cfg.NLayer), int32(cfg.NEmbd), int32(cfg.BlockSize), int32(cfg.NHead)} {
		if err := binary.Write(w, order, v); err != nil {
			return err
		}
	}

	// Tokenizer: numRunes + numBytes + UTF-8 encoded runes
	utf8Chars := []byte(string(tok.chars))
	if err := binary.Write(w, order, int32(len(tok.chars))); err != nil {
		return err
	}
	if err := binary.Write(w, order, int32(len(utf8Chars))); err != nil {
		return err
	}
	if _, err := w.Write(utf8Chars); err != nil {
		return err
	}

	// Training state
	if err := binary.Write(w, order, int32(step)); err != nil {
		return err
	}

	// Parameters
	if err := binary.Write(w, order, int32(len(params.data))); err != nil {
		return err
	}
	if err := binary.Write(w, order, params.data); err != nil {
		return err
	}
	if err := binary.Write(w, order, params.m); err != nil {
		return err
	}
	if err := binary.Write(w, order, params.v); err != nil {
		return err
	}

	return w.Flush()
}

const maxCheckpointAlloc = 1 << 30 // 1 GB sanity limit

func loadCheckpoint(path string) (*Params, *Tokenizer, *Config, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	order := binary.LittleEndian

	// Header
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read magic: %w", err)
	}
	if magic != checkpointMagic {
		return nil, nil, nil, 0, fmt.Errorf("not a microgpt checkpoint")
	}
	var version uint32
	if err := binary.Read(r, order, &version); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read version: %w", err)
	}
	if version != checkpointVersion {
		return nil, nil, nil, 0, fmt.Errorf("unsupported checkpoint version %d (want %d)", version, checkpointVersion)
	}

	// Hyperparameters
	var vals [4]int32
	if err := binary.Read(r, order, &vals); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read hyperparams: %w", err)
	}
	cfg := &Config{
		NLayer:    int(vals[0]),
		NEmbd:     int(vals[1]),
		BlockSize: int(vals[2]),
		NHead:     int(vals[3]),
	}
	if cfg.NLayer <= 0 || cfg.NEmbd <= 0 || cfg.BlockSize <= 0 || cfg.NHead <= 0 {
		return nil, nil, nil, 0, fmt.Errorf("invalid hyperparams: layers=%d embd=%d ctx=%d heads=%d",
			cfg.NLayer, cfg.NEmbd, cfg.BlockSize, cfg.NHead)
	}
	if cfg.NEmbd%cfg.NHead != 0 {
		return nil, nil, nil, 0, fmt.Errorf("embd (%d) not divisible by heads (%d)", cfg.NEmbd, cfg.NHead)
	}
	cfg.HeadDim = cfg.NEmbd / cfg.NHead

	// Tokenizer: numRunes + numBytes + UTF-8 encoded runes
	var numRunes, numBytes int32
	if err := binary.Read(r, order, &numRunes); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read numRunes: %w", err)
	}
	if err := binary.Read(r, order, &numBytes); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read numBytes: %w", err)
	}
	if numBytes < 0 || int64(numBytes) > maxCheckpointAlloc {
		return nil, nil, nil, 0, fmt.Errorf("invalid numBytes: %d", numBytes)
	}
	utf8Data := make([]byte, numBytes)
	if _, err := io.ReadFull(r, utf8Data); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read chars: %w", err)
	}
	chars := []rune(string(utf8Data))
	charToID := make(map[rune]int, numRunes)
	for i, ch := range chars {
		charToID[ch] = i
	}
	tok := &Tokenizer{
		chars:     chars,
		charToID:  charToID,
		bos:       int(numRunes),
		vocabSize: int(numRunes) + 1,
	}

	// Training state
	var step int32
	if err := binary.Read(r, order, &step); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read step: %w", err)
	}

	// Parameters
	var numParams int32
	if err := binary.Read(r, order, &numParams); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read numParams: %w", err)
	}
	if numParams < 0 || int64(numParams)*8*3 > maxCheckpointAlloc {
		return nil, nil, nil, 0, fmt.Errorf("invalid numParams: %d", numParams)
	}
	n := int(numParams)
	data := make([]float64, n)
	m := make([]float64, n)
	v := make([]float64, n)
	if err := binary.Read(r, order, data); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read data: %w", err)
	}
	if err := binary.Read(r, order, m); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read m: %w", err)
	}
	if err := binary.Read(r, order, v); err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read v: %w", err)
	}

	return &Params{data: data, m: m, v: v}, tok, cfg, int(step), nil
}

// ---------------------------------------------------------------------------
// Data loading & main
// ---------------------------------------------------------------------------

func loadDocs(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var docs []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024) // support lines up to 1 MB
	for sc.Scan() {
		if line := strings.TrimSpace(sc.Text()); line != "" {
			docs = append(docs, line)
		}
	}
	return docs, sc.Err()
}

func main() {
	var cfg Config
	var numSteps, numSamples, batchSize int
	var lr, temperature float64
	var dataFile, saveFile, loadFile string
	var seedFlag int64

	flag.IntVar(&cfg.NLayer, "layers", 1, "number of transformer layers")
	flag.IntVar(&cfg.NEmbd, "embd", 16, "embedding dimension")
	flag.IntVar(&cfg.BlockSize, "ctx", 16, "context window size")
	flag.IntVar(&cfg.NHead, "heads", 4, "number of attention heads")
	flag.IntVar(&numSteps, "steps", 5000, "number of training steps")
	flag.Float64Var(&lr, "lr", 0.01, "learning rate")
	flag.Float64Var(&temperature, "temp", 0.5, "sampling temperature")
	flag.IntVar(&numSamples, "samples", 20, "number of generated samples")
	flag.StringVar(&dataFile, "data", "input.txt", "path to dataset file")
	flag.Int64Var(&seedFlag, "seed", 0, "random seed (0 = random)")
	flag.StringVar(&saveFile, "save", "", "save checkpoint to file after training")
	flag.StringVar(&loadFile, "load", "", "load checkpoint from file before training")
	flag.IntVar(&batchSize, "batch", 1, "batch size for data-parallel training (documents per step)")
	flag.Parse()

	if batchSize < 1 {
		fmt.Fprintf(os.Stderr, "error: -batch must be >= 1\n")
		os.Exit(1)
	}

	var rngState, rngSeq uint64
	if seedFlag != 0 {
		rngState = uint64(seedFlag)
		rngSeq = 1
		fmt.Printf("seed: %d\n", seedFlag)
	} else {
		rngState, rngSeq = rand.Uint64(), rand.Uint64()
	}
	rng := rand.New(rand.NewPCG(rngState, rngSeq))

	var (
		params         *Params
		tok            *Tokenizer
		lastStep       int
		docs           []string
		trainStartStep int
	)

	if loadFile != "" {
		var (
			loadedCfg *Config
			err       error
		)
		params, tok, loadedCfg, lastStep, err = loadCheckpoint(loadFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load checkpoint: %v\n", err)
			os.Exit(1)
		}
		cfg = *loadedCfg
		fmt.Printf("loaded checkpoint: %s (step %d)\n", loadFile, lastStep)
		fmt.Printf("vocab size: %d | num params: %d\n", tok.vocabSize, len(params.data))
	} else {
		if cfg.NEmbd%cfg.NHead != 0 {
			fmt.Fprintf(os.Stderr, "error: -embd (%d) must be divisible by -heads (%d)\n", cfg.NEmbd, cfg.NHead)
			os.Exit(1)
		}
		cfg.HeadDim = cfg.NEmbd / cfg.NHead

		var err error
		docs, err = loadDocs(dataFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load: %v\n", err)
			os.Exit(1)
		}
		rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })
		fmt.Printf("num docs: %d\n", len(docs))

		tok = NewTokenizer(docs)
		fmt.Printf("vocab size: %d\n", tok.vocabSize)

		params = NewParams(&cfg, tok.vocabSize, rng)
		fmt.Printf("num params: %d\n", len(params.data))
	}

	// Training
	if numSteps > 0 {
		if docs == nil {
			var err error
			docs, err = loadDocs(dataFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "load docs: %v\n", err)
				os.Exit(1)
			}
		}
		tokenized := make([][]int, len(docs))
		for i, doc := range docs {
			tokenized[i] = tok.Encode(doc)
		}

		trainStartStep = lastStep

		// Epoch-based shuffle: every word is seen exactly once per epoch.
		perm := make([]int, len(tokenized))
		for j := range perm {
			perm[j] = j
		}

		const maxConsecutiveSkips = 10
		var lossRing [100]float64
		var lossSum float64
		var skipped, consecutiveSkips int
		var etaStr string
		trainStart := time.Now()
		etaUpdated := trainStart
		stepsRan := numSteps

		if batchSize == 1 {
			// Single-document path: identical to original behaviour.
			tape := NewTape(tapeInitCap)
			for i := range numSteps {
				step := lastStep + i
				j := i % len(tokenized)
				if j == 0 {
					rng.Shuffle(len(perm), func(a, b int) { perm[a], perm[b] = perm[b], perm[a] })
				}
				tokens := tokenized[perm[j]]
				lrT := computeLR(lr, step-trainStartStep, numSteps)
				loss := trainStep(params, &cfg, tok.vocabSize, tokens, step, lrT, tape)
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					skipped++
					consecutiveSkips++
					if consecutiveSkips >= maxConsecutiveSkips {
						fmt.Printf("\n%d consecutive bad batches — gradient explosion, stopping early\n", consecutiveSkips)
						stepsRan = i - consecutiveSkips + 1
						break
					}
					continue
				}
				consecutiveSkips = 0
				lossSum += loss
				if i >= 100 {
					lossSum -= lossRing[i%100]
				}
				lossRing[i%100] = loss
				window := min(i+1, 100)
				now := time.Now()
				if i >= 10000 && now.Sub(etaUpdated) >= time.Second {
					avgStep := now.Sub(trainStart) / time.Duration(i+1)
					remaining := avgStep * time.Duration(numSteps-i-1)
					etaStr = " | eta " + remaining.Round(time.Second).String()
					etaUpdated = now
				}
				fmt.Printf("\rstep %4d / %4d | loss %.4f | avg100 %.4f%s\x1b[K", i+1, numSteps, loss, lossSum/float64(window), etaStr)
			}
		} else {
			// Batched data-parallel path.
			workers := newTrainWorkers(batchSize, &cfg)
			tokensBatch := make([][]int, batchSize)
			j := 0 // cursor into perm
			for i := range numSteps {
				step := lastStep + i
				// Fill batch, reshuffling at epoch boundaries.
				for b := range batchSize {
					if j%len(tokenized) == 0 {
						rng.Shuffle(len(perm), func(a, c int) { perm[a], perm[c] = perm[c], perm[a] })
						j = 0
					}
					tokensBatch[b] = tokenized[perm[j%len(tokenized)]]
					j++
				}
				lrT := computeLR(lr, step-trainStartStep, numSteps)
				loss := trainStepBatch(params, &cfg, tok.vocabSize, tokensBatch, step, lrT, workers)
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					skipped++
					consecutiveSkips++
					if consecutiveSkips >= maxConsecutiveSkips {
						fmt.Printf("\n%d consecutive bad batches — gradient explosion, stopping early\n", consecutiveSkips)
						stepsRan = i - consecutiveSkips + 1
						break
					}
					continue
				}
				consecutiveSkips = 0
				lossSum += loss
				if i >= 100 {
					lossSum -= lossRing[i%100]
				}
				lossRing[i%100] = loss
				window := min(i+1, 100)
				now := time.Now()
				if i >= 10000 && now.Sub(etaUpdated) >= time.Second {
					avgStep := now.Sub(trainStart) / time.Duration(i+1)
					remaining := avgStep * time.Duration(numSteps-i-1)
					etaStr = " | eta " + remaining.Round(time.Second).String()
					etaUpdated = now
				}
				fmt.Printf("\rstep %4d / %4d | loss %.4f | avg100 %.4f%s\x1b[K", i+1, numSteps, loss, lossSum/float64(window), etaStr)
			}
		}
		trainElapsed := time.Since(trainStart)
		numSteps = stepsRan
		if skipped > 0 {
			fmt.Printf("\nskipped batches:  %d (bad gradients, params preserved)\n", skipped)
		}

		fmt.Printf("\n--- training stats ---\n")
		fmt.Printf("total time:    %v\n", trainElapsed.Round(time.Millisecond))
		fmt.Printf("avg step:      %v\n", (trainElapsed / time.Duration(numSteps)).Round(time.Microsecond))
		fmt.Printf("throughput:    %.1f steps/sec\n", float64(numSteps)/trainElapsed.Seconds())
		if batchSize > 1 {
			fmt.Printf("batch size:    %d (data-parallel)\n", batchSize)
			docsTotal := int64(numSteps) * int64(batchSize)
			fmt.Printf("docs/sec:      %.1f\n", float64(docsTotal)/trainElapsed.Seconds())
		}

		lastStep += numSteps
	}

	// Save checkpoint
	if saveFile != "" {
		if err := saveCheckpoint(saveFile, params, tok, &cfg, lastStep); err != nil {
			fmt.Fprintf(os.Stderr, "save checkpoint: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("checkpoint saved: %s (step %d)\n", saveFile, lastStep)
	}

	// Inference
	fmt.Println("--- inference ---")
	inferStart := time.Now()
	for range numSamples {
		name := generate(params, tok, &cfg, temperature, rng)
		fmt.Println(name)
	}
	inferElapsed := time.Since(inferStart)
	fmt.Printf("--- inference stats ---\n")
	fmt.Printf("total time:    %v\n", inferElapsed.Round(time.Millisecond))
	fmt.Printf("avg sample:    %v\n", (inferElapsed / time.Duration(numSamples)).Round(time.Microsecond))
}
