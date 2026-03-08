package main

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
)

// testConfig returns a Config for use in tests — no global state mutation.
func testConfig() *Config {
	return &Config{
		NLayer:    1,
		NEmbd:     16,
		BlockSize: 16,
		NHead:     4,
		HeadDim:   4,
	}
}

// --- Tape op gradient checks ---

// numericalGrad computes (f(x+eps) - f(x-eps)) / (2*eps) for a single leaf.
func numericalGrad(build func(t *Tape, leaf Idx) Idx, x, eps float64) float64 {
	t1 := NewTape(1024)
	a1 := t1.Leaf(x + eps)
	out1 := build(t1, a1)

	t2 := NewTape(1024)
	a2 := t2.Leaf(x - eps)
	out2 := build(t2, a2)

	return (t1.Val(out1) - t2.Val(out2)) / (2 * eps)
}

func checkGrad(t *testing.T, name string, build func(tp *Tape, leaf Idx) Idx, x float64) {
	t.Helper()
	const eps = 1e-5

	tp := NewTape(1024)
	leaf := tp.Leaf(x)
	out := build(tp, leaf)
	tp.Backward(out)
	analytical := tp.Grad(leaf)

	numerical := numericalGrad(build, x, eps)

	if diff := math.Abs(analytical - numerical); diff > 1e-4 {
		t.Errorf("%s: analytical=%.8f numerical=%.8f diff=%.8f", name, analytical, numerical, diff)
	}
}

func TestTapeOps(t *testing.T) {
	// Add
	checkGrad(t, "Add(x, 3)", func(tp *Tape, leaf Idx) Idx {
		return tp.Add(leaf, tp.Leaf(3))
	}, 2.0)

	// Mul
	checkGrad(t, "Mul(x, 5)", func(tp *Tape, leaf Idx) Idx {
		return tp.Mul(leaf, tp.Leaf(5))
	}, 3.0)

	// Pow
	checkGrad(t, "Pow(x, 3)", func(tp *Tape, leaf Idx) Idx {
		return tp.Pow(leaf, 3)
	}, 2.0)

	// Log
	checkGrad(t, "Log(x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Log(leaf)
	}, 2.0)

	// Exp
	checkGrad(t, "Exp(x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Exp(leaf)
	}, 1.0)

	// ReLU positive
	checkGrad(t, "ReLU(x>0)", func(tp *Tape, leaf Idx) Idx {
		return tp.ReLU(leaf)
	}, 2.0)

	// ReLU negative
	checkGrad(t, "ReLU(x<0)", func(tp *Tape, leaf Idx) Idx {
		return tp.ReLU(leaf)
	}, -2.0)

	// Sub
	checkGrad(t, "Sub(x, 3)", func(tp *Tape, leaf Idx) Idx {
		return tp.Sub(leaf, tp.Leaf(3))
	}, 2.0)

	// Neg
	checkGrad(t, "Neg(x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Neg(leaf)
	}, 5.0)

	// Div
	checkGrad(t, "Div(x, 3)", func(tp *Tape, leaf Idx) Idx {
		return tp.Div(leaf, tp.Leaf(3))
	}, 6.0)

	// Sum
	checkGrad(t, "Sum([x, 2, 3])", func(tp *Tape, leaf Idx) Idx {
		return tp.Sum([]Idx{leaf, tp.Leaf(2), tp.Leaf(3)})
	}, 1.0)

	// Second operand gradients for binary ops
	checkGrad(t, "Sub(3, x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Sub(tp.Leaf(3), leaf)
	}, 2.0)

	checkGrad(t, "Div(6, x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Div(tp.Leaf(6), leaf)
	}, 3.0)

	checkGrad(t, "Mul(5, x)", func(tp *Tape, leaf Idx) Idx {
		return tp.Mul(tp.Leaf(5), leaf)
	}, 3.0)

	// Dot product (first operand)
	checkGrad(t, "Dot([x,2,3],[4,5,6])", func(tp *Tape, leaf Idx) Idx {
		a := []Idx{leaf, tp.Leaf(2), tp.Leaf(3)}
		b := []Idx{tp.Leaf(4), tp.Leaf(5), tp.Leaf(6)}
		return tp.Dot(a, b)
	}, 1.0)

	// Dot product (second operand)
	checkGrad(t, "Dot([4,5,6],[x,2,3])", func(tp *Tape, leaf Idx) Idx {
		a := []Idx{tp.Leaf(4), tp.Leaf(5), tp.Leaf(6)}
		b := []Idx{leaf, tp.Leaf(2), tp.Leaf(3)}
		return tp.Dot(a, b)
	}, 1.0)

	// Compound: softmax-like
	checkGrad(t, "Exp(x)/Sum(Exp)", func(tp *Tape, leaf Idx) Idx {
		a := tp.Leaf(1.0)
		b := leaf
		c := tp.Leaf(3.0)
		ea := tp.Exp(a)
		eb := tp.Exp(b)
		ec := tp.Exp(c)
		total := tp.Sum([]Idx{ea, eb, ec})
		return tp.Div(eb, total)
	}, 2.0)
}

// --- Deterministic training test ---

func TestTrainStepDeterministic(t *testing.T) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	rng.Shuffle(len(docs), func(i, j int) { docs[i], docs[j] = docs[j], docs[i] })

	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	tokenized := make([][]int, len(docs))
	for i, doc := range docs {
		tokenized[i] = tok.Encode(doc)
	}

	numSteps := 5

	// Run 5 training steps and verify loss is finite and deterministic
	losses := make([]float64, numSteps)
	for step := range numSteps {
		tokens := tokenized[step%len(tokenized)]
		lrT := computeLR(0.01, step, numSteps)
		losses[step] = trainStep(params, cfg, tok.vocabSize, tokens, step, lrT, nil)
	}

	for i, l := range losses {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("step %d: loss is %v", i, l)
		}
	}

	// Run again with same seed — must produce identical results
	rng2 := rand.New(rand.NewPCG(42, 0))
	docs2 := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	rng2.Shuffle(len(docs2), func(i, j int) { docs2[i], docs2[j] = docs2[j], docs2[i] })

	tok2 := NewTokenizer(docs2)
	params2 := NewParams(cfg, tok2.vocabSize, rng2)

	tokenized2 := make([][]int, len(docs2))
	for i, doc := range docs2 {
		tokenized2[i] = tok2.Encode(doc)
	}

	for step := range numSteps {
		tokens := tokenized2[step%len(tokenized2)]
		lrT := computeLR(0.01, step, numSteps)
		loss := trainStep(params2, cfg, tok2.vocabSize, tokens, step, lrT, nil)
		if loss != losses[step] {
			t.Fatalf("step %d: expected loss %.10f, got %.10f", step, losses[step], loss)
		}
	}
}

// --- Tape reuse test ---

func TestTapeReuse(t *testing.T) {
	cfg := testConfig()

	docs := []string{"emma", "olivia", "ava"}
	tok := NewTokenizer(docs)

	// Run with fresh tape (nil)
	rng1 := rand.New(rand.NewPCG(42, 0))
	params1 := NewParams(cfg, tok.vocabSize, rng1)
	tokens := tok.Encode("emma")
	lrT := computeLR(0.01, 0, 5)
	loss1 := trainStep(params1, cfg, tok.vocabSize, tokens, 0, lrT, nil)

	// Run with reused tape
	rng2 := rand.New(rand.NewPCG(42, 0))
	params2 := NewParams(cfg, tok.vocabSize, rng2)
	tape := NewTape(tapeInitCap)
	_ = trainStep(params2, cfg, tok.vocabSize, tokens, 0, lrT, tape) // first use
	// Run again on the reused tape with fresh params
	rng3 := rand.New(rand.NewPCG(42, 0))
	params3 := NewParams(cfg, tok.vocabSize, rng3)
	loss2 := trainStep(params3, cfg, tok.vocabSize, tokens, 0, lrT, tape)

	if loss1 != loss2 {
		t.Fatalf("tape reuse mismatch: fresh=%.10f reused=%.10f", loss1, loss2)
	}
}

// --- Inference matches training forward pass ---

func TestInferenceMatchesTraining(t *testing.T) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	// Tape-based forward
	tp := NewTape(tapeInitCap)
	m, _ := loadModel(tp, params, cfg, tok.vocabSize)
	kv1 := newKVCache(cfg.NLayer)
	ws := newFwdWorkspace(cfg)
	tapeLogits := gptForward(tp, &m, cfg, tok.bos, 0, kv1, ws)

	// Tape-free forward
	im := loadInferModel(params, cfg, tok.vocabSize)
	kv2 := &inferKV{
		keys:   make([][][]float64, cfg.NLayer),
		values: make([][][]float64, cfg.NLayer),
	}
	iws := newInferWorkspace(cfg, tok.vocabSize)
	gptForwardF64(&im, cfg, tok.bos, 0, kv2, iws)

	for i := range iws.logits {
		tapeVal := tp.Val(tapeLogits[i])
		if math.Abs(tapeVal-iws.logits[i]) > 1e-12 {
			t.Errorf("logit[%d]: tape=%.15f f64=%.15f", i, tapeVal, iws.logits[i])
		}
	}
}

// --- Checkpoint roundtrip ---

func TestCheckpointRoundtrip(t *testing.T) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	// Run a few steps so Adam buffers are non-zero
	tape := NewTape(tapeInitCap)
	tokens := tok.Encode("emma")
	for step := range 3 {
		lrT := computeLR(0.01, step, 5)
		trainStep(params, cfg, tok.vocabSize, tokens, step, lrT, tape)
	}
	savedStep := 3

	// Save
	path := t.TempDir() + "/test.bin"
	if err := saveCheckpoint(path, params, tok, cfg, savedStep); err != nil {
		t.Fatalf("saveCheckpoint: %v", err)
	}

	// Load
	params2, tok2, cfg2, step2, err := loadCheckpoint(path)
	if err != nil {
		t.Fatalf("loadCheckpoint: %v", err)
	}

	// Verify step
	if step2 != savedStep {
		t.Errorf("step: got %d, want %d", step2, savedStep)
	}

	// Verify hyperparams restored
	if cfg2.NLayer != 1 || cfg2.NEmbd != 16 || cfg2.BlockSize != 16 || cfg2.NHead != 4 {
		t.Errorf("hyperparams mismatch after load: %+v", cfg2)
	}

	// Verify tokenizer
	if tok2.vocabSize != tok.vocabSize {
		t.Errorf("vocabSize: got %d, want %d", tok2.vocabSize, tok.vocabSize)
	}
	for i, ch := range tok.chars {
		if tok2.chars[i] != ch {
			t.Errorf("chars[%d]: got %q, want %q", i, tok2.chars[i], ch)
		}
	}

	// Verify params
	if len(params2.data) != len(params.data) {
		t.Fatalf("data length mismatch: %d vs %d", len(params2.data), len(params.data))
	}
	for i := range params.data {
		if params2.data[i] != params.data[i] {
			t.Errorf("data[%d]: got %v, want %v", i, params2.data[i], params.data[i])
		}
		if params2.m[i] != params.m[i] {
			t.Errorf("m[%d]: got %v, want %v", i, params2.m[i], params.m[i])
		}
		if params2.v[i] != params.v[i] {
			t.Errorf("v[%d]: got %v, want %v", i, params2.v[i], params.v[i])
		}
	}

	// Verify continued training gives same result from loaded checkpoint
	tokens2 := tok2.Encode("olivia")
	lrT := computeLR(0.01, savedStep, 5)
	loss1 := trainStep(params, cfg, tok.vocabSize, tokens2, savedStep, lrT, tape)
	loss2 := trainStep(params2, cfg2, tok2.vocabSize, tokens2, savedStep, lrT, tape)
	if loss1 != loss2 {
		t.Errorf("loss after resume: orig=%.10f loaded=%.10f", loss1, loss2)
	}
}

// --- Edge-case tests ---

func TestTokenizerSingleChar(t *testing.T) {
	docs := []string{"a", "b", "c"}
	tok := NewTokenizer(docs)

	if tok.vocabSize != 4 { // a, b, c + BOS
		t.Fatalf("vocabSize: got %d, want 4", tok.vocabSize)
	}

	encoded := tok.Encode("abc")
	if encoded[0] != tok.bos || encoded[len(encoded)-1] != tok.bos {
		t.Errorf("Encode missing BOS: %v", encoded)
	}

	// Roundtrip
	var sb strings.Builder
	for _, id := range encoded[1 : len(encoded)-1] {
		if ch, ok := tok.Decode(id); ok {
			sb.WriteRune(ch)
		}
	}
	if sb.String() != "abc" {
		t.Errorf("roundtrip: got %q, want %q", sb.String(), "abc")
	}
}

func TestGenerateDoesNotPanic(t *testing.T) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"hello", "world"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	// Should complete without panic
	_ = generate(params, tok, cfg, 0.5, rng)
}

func TestLoadDocsLongLine(t *testing.T) {
	// Write temp file with 100KB line
	path := t.TempDir() + "/long.txt"
	longLine := strings.Repeat("a", 100*1024)
	if err := os.WriteFile(path, []byte(longLine+"\n"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	docs, err := loadDocs(path)
	if err != nil {
		t.Fatalf("loadDocs: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if len(docs[0]) != 100*1024 {
		t.Errorf("doc length: got %d, want %d", len(docs[0]), 100*1024)
	}
}

func TestCheckpointInvalidData(t *testing.T) {
	// Bad magic
	path := t.TempDir() + "/bad.bin"
	if err := os.WriteFile(path, []byte("NOTAMAGIC"), 0644); err != nil {
		t.Fatal(err)
	}
	_, _, _, _, err := loadCheckpoint(path)
	if err == nil {
		t.Error("expected error for bad magic")
	}

	// Valid header but huge numParams
	path2 := t.TempDir() + "/huge.bin"
	f, err := os.Create(path2)
	if err != nil {
		t.Fatal(err)
	}
	order := binary.LittleEndian
	f.Write(checkpointMagic[:])
	binary.Write(f, order, checkpointVersion)
	// Hyperparams: nLayer=1, nEmbd=16, blockSize=16, nHead=4
	for _, v := range []int32{1, 16, 16, 4} {
		binary.Write(f, order, v)
	}
	// Tokenizer: 3 runes, 3 bytes
	binary.Write(f, order, int32(3))
	binary.Write(f, order, int32(3))
	f.Write([]byte("abc"))
	// Step
	binary.Write(f, order, int32(0))
	// Huge numParams
	binary.Write(f, order, int32(1<<30))
	f.Close()

	_, _, _, _, err = loadCheckpoint(path2)
	if err == nil {
		t.Error("expected error for huge numParams")
	}

	// Bad nEmbd % nHead
	path3 := t.TempDir() + "/badmod.bin"
	f, err = os.Create(path3)
	if err != nil {
		t.Fatal(err)
	}
	f.Write(checkpointMagic[:])
	binary.Write(f, order, checkpointVersion)
	// nEmbd=15, nHead=4 → 15%4 != 0
	for _, v := range []int32{1, 15, 16, 4} {
		binary.Write(f, order, v)
	}
	f.Close()

	_, _, _, _, err = loadCheckpoint(path3)
	if err == nil {
		t.Error("expected error for nEmbd%%nHead != 0")
	}
}

func TestComputeLR(t *testing.T) {
	// At step 0, warmup phase: LR should be 0
	lr := computeLR(0.01, 0, 100)
	if lr != 0 {
		t.Errorf("step 0: expected 0, got %f", lr)
	}

	// At last step, LR should be 0
	lr = computeLR(0.01, 100, 100)
	if lr != 0 {
		t.Errorf("last step: expected 0, got %f", lr)
	}

	// Mid training, LR should be positive
	lr = computeLR(0.01, 50, 100)
	if lr <= 0 {
		t.Errorf("mid training: expected positive LR, got %f", lr)
	}
}

// --- Benchmarks ---

func BenchmarkTrainStep(b *testing.B) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	tokenized := make([][]int, len(docs))
	for i, doc := range docs {
		tokenized[i] = tok.Encode(doc)
	}

	tape := NewTape(tapeInitCap)
	b.ResetTimer()
	for i := range b.N {
		tokens := tokenized[i%len(tokenized)]
		lrT := computeLR(0.01, i%100, 100)
		trainStep(params, cfg, tok.vocabSize, tokens, i, lrT, tape)
	}
}

func BenchmarkGenerate(b *testing.B) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	b.ResetTimer()
	for range b.N {
		generate(params, tok, cfg, 0.5, rng)
	}
}

// --- Batch training tests ---

// TestTrainStepBatchSingleEquivalence verifies that trainStepBatch with batch=1
// produces identical results to trainStep for the same document.
func TestTrainStepBatchSingleEquivalence(t *testing.T) {
	cfg := testConfig()

	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}

	for step := range 3 {
		// Run trainStep
		rng1 := rand.New(rand.NewPCG(42, 0))
		tok1 := NewTokenizer(docs)
		params1 := NewParams(cfg, tok1.vocabSize, rng1)
		tokens := tok1.Encode(docs[step%len(docs)])
		lrT := computeLR(0.01, step, 10)
		loss1 := trainStep(params1, cfg, tok1.vocabSize, tokens, step, lrT, nil)

		// Run trainStepBatch with batch=1
		rng2 := rand.New(rand.NewPCG(42, 0))
		tok2 := NewTokenizer(docs)
		params2 := NewParams(cfg, tok2.vocabSize, rng2)
		tokens2 := tok2.Encode(docs[step%len(docs)])
		workers := newTrainWorkers(1, cfg)
		loss2 := trainStepBatch(params2, cfg, tok2.vocabSize, [][]int{tokens2}, step, lrT, workers)

		if loss1 != loss2 {
			t.Fatalf("step %d: trainStep loss=%.10f != trainStepBatch loss=%.10f", step, loss1, loss2)
		}
		// Verify parameters are identical
		for i := range params1.data {
			if params1.data[i] != params2.data[i] {
				t.Fatalf("step %d: param[%d] differs: %.10f vs %.10f", step, i, params1.data[i], params2.data[i])
			}
		}
	}
}

// TestTrainStepBatchDeterministic verifies that trainStepBatch with batch>1
// produces identical results across runs with the same seed.
func TestTrainStepBatchDeterministic(t *testing.T) {
	cfg := testConfig()
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}

	run := func() []float64 {
		rng := rand.New(rand.NewPCG(99, 0))
		tok := NewTokenizer(docs)
		params := NewParams(cfg, tok.vocabSize, rng)
		tokenized := make([][]int, len(docs))
		for i, doc := range docs {
			tokenized[i] = tok.Encode(doc)
		}
		workers := newTrainWorkers(2, cfg)
		losses := make([]float64, 5)
		for step := range 5 {
			batch := [][]int{
				tokenized[step%len(tokenized)],
				tokenized[(step+1)%len(tokenized)],
			}
			lrT := computeLR(0.01, step, 5)
			losses[step] = trainStepBatch(params, cfg, tok.vocabSize, batch, step, lrT, workers)
		}
		return losses
	}

	losses1 := run()
	losses2 := run()
	for i := range losses1 {
		if losses1[i] != losses2[i] {
			t.Fatalf("step %d: loss %.10f != %.10f", i, losses1[i], losses2[i])
		}
	}
}

// TestTrainStepBatchQuality verifies that batch training reduces loss over time.
func TestTrainStepBatchQuality(t *testing.T) {
	cfg := testConfig()
	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)
	tokenized := make([][]int, len(docs))
	for i, doc := range docs {
		tokenized[i] = tok.Encode(doc)
	}

	workers := newTrainWorkers(2, cfg)
	numSteps := 100
	var firstLoss, lastLoss float64
	for step := range numSteps {
		batch := [][]int{
			tokenized[step%len(tokenized)],
			tokenized[(step+1)%len(tokenized)],
		}
		lrT := computeLR(0.01, step, numSteps)
		loss := trainStepBatch(params, cfg, tok.vocabSize, batch, step, lrT, workers)
		if step == 0 {
			firstLoss = loss
		}
		if step == numSteps-1 {
			lastLoss = loss
		}
	}
	if lastLoss >= firstLoss {
		t.Fatalf("loss did not decrease: first=%.4f last=%.4f", firstLoss, lastLoss)
	}
}

func BenchmarkTrainStepBatch(b *testing.B) {
	cfg := testConfig()

	rng := rand.New(rand.NewPCG(42, 0))
	docs := []string{"emma", "olivia", "ava", "sophia", "isabella"}
	tok := NewTokenizer(docs)
	params := NewParams(cfg, tok.vocabSize, rng)

	tokenized := make([][]int, len(docs))
	for i, doc := range docs {
		tokenized[i] = tok.Encode(doc)
	}

	workers := newTrainWorkers(4, cfg)
	batch := make([][]int, 4)
	b.ResetTimer()
	for i := range b.N {
		for j := range 4 {
			batch[j] = tokenized[(i*4+j)%len(tokenized)]
		}
		lrT := computeLR(0.01, i%100, 100)
		trainStepBatch(params, cfg, tok.vocabSize, batch, i, lrT, workers)
	}
}
