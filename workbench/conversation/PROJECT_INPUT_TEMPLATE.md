# Single consolidated project-input request

Use this only after read-only discovery fails to locate essential inputs. Ask once:

```text
为直接开始，请在一条消息中提供或确认以下内容（已上传的项目可写“使用附件”，已知路径可写“沿用包内路径”）：
1. VLA 源码/仓库路径；
2. checkpoint 路径；
3. 待测加速模块源码或附件；
4. 任务套件、任务范围和成功率协议；
5. 云端 Python、GPU 与实验输出根目录。
如果只需要分析现有 JSON/log，请直接上传文件，不需要填写其余项目。
```

Do not ask separately for each field. If the task is result analysis, source review, or
another read-only action that can proceed without all five inputs, proceed with what is
available.

## User shortcuts the Codex should understand

- `分析附件并决定下一步` — parse the attached result/log, report gates, then decide
  expand/tune/reject.
- `评估附件里的新模块` — audit the attached source and identify the real integration
  point before writing code.
- `沿用包内路径` — use the recorded cloud paths as hypotheses, verify them before use,
  and do not ask the user to retype them.
- `生成云端实验包` — create and validate one complete ZIP plus one pasteable command
  block; do not return scattered scripts.

