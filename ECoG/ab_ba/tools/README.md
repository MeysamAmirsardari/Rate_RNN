# MATLAB figure reference extractor

`ExtractFigData.java` extracts the actual plotted `Line` and `Bar` arrays from
the modern MCOS objects in the six supplied `.fig` files. It is a reference
audit only; it does not replace recomputation from labeled epochs.

The utility uses the open-source HEBI Robotics MAT File Library, MFL 0.5.15.
The dependency is intentionally not copied into the repository. After obtaining
`mfl-core-0.5.15.jar` from Maven Central:

```bash
javac -cp /path/to/mfl-core-0.5.15.jar ExtractFigData.java
java -cp .:/path/to/mfl-core-0.5.15.jar \
  /path/to/reference.fig /path/to/reference.tsv
```

Every TSV row retains its MCOS object path, display name, sample number, and
exact double-precision x/y values. Duplicate graph-object references are
removed by object identity.
