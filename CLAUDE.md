# LiteRT LM Fork (Custom Engine)
    
## 🚧 Critical Architecture Constraints 🚧

This fork of `litert` modifies the core engine to support advanced features such as **KV Cache Cloning / Snapshotting** and **Prefix Matching Fast-Forward**. 

When working in this directory (especially inside `runtime/framework/resource_management/` and `runtime/executor/`), **YOU MUST OBEY THE FOLLOWING ARCHITECTURAL BOUNDARIES**:

### 1. Engine is Abstract; Do not mix App Logic into C++
- The C++ engine should **NEVER** know anything about our Android app's specific features like "Barge-in", "VAD", "Draft / Main sessions", or "StatefulDialogManager".
- All C++ logic must be generalized. We are building primitive features (like memory snapshotting and step manipulation) that are useful to the broader LLM community (e.g. for speculative decoding or parallel prompting).
- If you find yourself adding variables with names like `is_barge_in` or `draft_mode` to C++, **STOP**. That logic belongs in the Kotlin app layer (`android/app/src/...`).

### 2. KV Cache Ownership Invariants
- `ContextHandler` acts as a facade, but when loaded into the executor, the `LlmExecutor` takes over full ownership of `RuntimeConfig`, `RuntimeState`, and `ProcessedContext`.
- If a session is suspended/switched, the state must be moved out of the `LlmExecutor` back into the `ContextHandler` using `RetrieveRuntimeConfig()`, etc. Do not leave pointers dangling. Look at `SaveProcessedContextAndSeparateLoadedHandler`.

### 3. Step Manipulation
- We allow `SetCurrentStep(new_step)` to advance forward (`new_step > old_step`) **ONLY IF** the new step is bounded by `TokenCount()` of the cached KV pairs.
- This represents our "Prefix Matching" optimization. It must remain general. Do not add hacky step resets.

By keeping this boundary clean, our fork remains modular, and we can seamlessly submit our engine changes as Pull Requests to upstream (Google / LiteRT) while keeping our voice assistant logic isolated in our proprietary app.
