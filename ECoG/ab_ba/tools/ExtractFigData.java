/*
 * Extract plotted numerical arrays from modern MATLAB .fig MCOS objects.
 *
 * Compile with the open-source HEBI Robotics MAT File Library (MFL):
 *   javac -cp mfl-core-0.5.15.jar ExtractFigData.java
 *   java -cp .:mfl-core-0.5.15.jar ExtractFigData input.fig output.tsv
 *
 * MFL is not vendored in this repository. This utility uses reflection only
 * because MATLAB's MCOS figure storage is undocumented and MFL intentionally
 * keeps its resolved object implementation package-private.
 */

import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.types.*;

import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Paths;
import java.util.*;

public final class ExtractFigData {
    private final Set<Array> seenArrays = Collections.newSetFromMap(new IdentityHashMap<>());
    private final Set<Object> seenObjects = Collections.newSetFromMap(new IdentityHashMap<>());
    private final PrintWriter output;
    private int nextObjectId = 1;

    private ExtractFigData(PrintWriter output) {
        this.output = output;
        output.println("object_id\tkind\tobject_path\tdisplay_name\tsample\tx\ty");
    }

    private static String text(Array value) {
        return value instanceof Char ? ((Char) value).getString() : "";
    }

    private static String clean(String value) {
        return value.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ');
    }

    private void emit(Object object, String className, List<String> fields,
                      Method get, String path) throws Exception {
        if (!className.equals("matlab.graphics.chart.primitive.Line")
                && !className.equals("matlab.graphics.chart.primitive.Bar")) return;
        Array xValue = null, yValue = null, displayValue = null;
        for (String field : fields) {
            Array value = (Array) get.invoke(object, field);
            if (field.equals("XData_I") || (xValue == null && field.equals("XData"))) xValue = value;
            if (field.equals("YData_I") || (yValue == null && field.equals("YData"))) yValue = value;
            if (field.equals("DisplayName_I")) displayValue = value;
        }
        if (!(xValue instanceof Matrix) || !(yValue instanceof Matrix)) return;
        Matrix x = (Matrix) xValue;
        Matrix y = (Matrix) yValue;
        if (x.getNumElements() != y.getNumElements()) return;
        if (className.endsWith("Line") && y.getNumElements() <= 100 && text(displayValue).isEmpty()) return;
        int objectId = nextObjectId++;
        String kind = className.substring(className.lastIndexOf('.') + 1);
        String display = clean(text(displayValue));
        for (int index = 0; index < y.getNumElements(); index++) {
            output.printf(Locale.ROOT, "%d\t%s\t%s\t%s\t%d\t%.17g\t%.17g%n",
                    objectId, kind, clean(path), display, index,
                    x.getDouble(index), y.getDouble(index));
        }
    }

    private void walk(Array value, String path, int depth) throws Exception {
        if (value == null || depth > 48 || !seenArrays.add(value)) return;
        if (value instanceof ObjectStruct) {
            try {
                Field objectsField = value.getClass().getDeclaredField("objects");
                objectsField.setAccessible(true);
                Object[] objects = (Object[]) objectsField.get(value);
                for (int index = 0; index < objects.length; index++) {
                    Object object = objects[index];
                    if (!seenObjects.add(object)) continue;
                    Method getPackage = object.getClass().getDeclaredMethod("getPackageName");
                    Method getClass = object.getClass().getDeclaredMethod("getClassName");
                    Method getFields = object.getClass().getDeclaredMethod("getFieldNames");
                    Method get = object.getClass().getDeclaredMethod("get", String.class);
                    getPackage.setAccessible(true);
                    getClass.setAccessible(true);
                    getFields.setAccessible(true);
                    get.setAccessible(true);
                    String className = getPackage.invoke(object) + "." + getClass.invoke(object);
                    @SuppressWarnings("unchecked")
                    List<String> fields = (List<String>) getFields.invoke(object);
                    String objectPath = path + "[" + index + "]";
                    emit(object, className, fields, get, objectPath);
                    for (String field : fields) {
                        Array child = (Array) get.invoke(object, field);
                        walk(child, objectPath + "." + field, depth + 1);
                    }
                }
                return;
            } catch (NoSuchFieldException ignored) {
                // Fall through to the public Struct interface.
            }
        }
        if (value instanceof Struct) {
            Struct struct = (Struct) value;
            for (String field : struct.getFieldNames()) {
                for (int index = 0; index < struct.getNumElements(); index++) {
                    walk(struct.get(field, index), path + "." + field + "[" + index + "]", depth + 1);
                }
            }
        } else if (value instanceof Cell) {
            Cell cell = (Cell) value;
            for (int index = 0; index < cell.getNumElements(); index++) {
                walk(cell.get(index), path + "{" + index + "}", depth + 1);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) throw new IllegalArgumentException("Usage: ExtractFigData input.fig output.tsv");
        try (MatFile file = Mat5.readFromFile(args[0]);
             PrintWriter output = new PrintWriter(Paths.get(args[1]).toFile(), "UTF-8")) {
            ExtractFigData extractor = new ExtractFigData(output);
            for (MatFile.Entry entry : file.getEntries()) {
                extractor.walk(entry.getValue(), entry.getName(), 0);
            }
        }
    }
}
