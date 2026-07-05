# nosql

`nosql` is an experiment in building a tiny text-to-SQL model.

The goal is to distill a larger teacher model into a much smaller student model, then quantize it as aggressively as possible. The final model should only do one task:

```txt
schema + question -> SQL
```

This project uses SQL because it is easy to validate and benchmark. A generated query can be checked for syntax, schema correctness, execution success, and result correctness.

The teacher model will likely be Qwen or Devstral. The first approach is response distillation: use the teacher to produce SQL examples, keep the valid ones, train the student model on them, then quantize the result.

The main experiment is whether a very small specialized model can outperform a larger general-purpose teacher on a narrow text-to-SQL benchmark.

Background notes are kept in `docs/`.

