# Evaluation protocol

## 1. Reproducibility record

Record GPU, driver, CUDA, Python, PyTorch, Transformers fork/commit, VLA commit, acceleration commit, checkpoint checksum, task/episode/seed, precision, attention backend, and every source change. Treat warnings about dependency-version drift as experimental metadata.

## 2. Runtime discovery

Load the unmodified model first. Capture the actual model class and named modules, then observe one real forward. Identify the vision output, projector output, decoder prefill, iterative decode, cache object, attention mask, position IDs, and action output. A checkpoint's configuration text is not sufficient evidence of runtime structure.

## 3. Adapter control

Provide three modes when possible: baseline, accelerated, and zero-effect control. The zero-effect control must exercise the adapter path while retaining all information. It should match baseline output and should reveal wrapper overhead.

For stateful methods, define lifecycle transitions: first frame, cached frame, reset, new episode, exception recovery, and model reload. Clear frame-dependent configuration before every call and repopulate it only from the current state.

## 4. Paired measurement

Use identical observations and instructions. Warm up both modes, alternate execution order, synchronize CUDA around timing, and report median and p95. Record end-to-end wall time plus vision, projector, cache/pruner, LLM prefill, and LLM decode stages. Also record memory and the actual kept/reused count.

## 5. Fidelity and behavior

Compare full action vectors with mean and maximum absolute error and per-dimension error. Report gripper sign agreement separately. Test deterministic repeatability, execution-order effects, changing observations, and known-success closed-loop episodes. Synthetic frames are smoke tests only.

## 6. Tuning discipline

Change one parameter family at a time. Keep the default configuration in every scan. Select a candidate using both latency and behavior; do not optimize only for token count. Re-run the selected configuration with more samples and closed-loop episodes.

## 7. Admission

Create a result manifest, validate it, and run `admission_decision.py`. Preserve raw JSON evidence beside the manifest. A PASS means this model, checkpoint, task distribution, hardware/software environment, adapter, and parameter set passed; it is not a universal claim.
