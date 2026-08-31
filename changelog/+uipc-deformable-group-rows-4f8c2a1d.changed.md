Store authored UIPC cloth and soft-body groups as `uipc:cloth` and
`uipc:deformable_body` custom-frequency rows. Call
`SolverUIPC.register_custom_attributes(builder)` before `finalize()` to retain
separate groups, labels, topology ranges, densities, and per-group
constitutions. Remove the public `ClothRange` / `SoftBodyRange` types and the
range return values from `ModelBuilder.add_cloth_*` / `add_soft_*`.
