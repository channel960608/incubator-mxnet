package org.apache.mxnet.jna;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.types.DataType;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.ndarray.types.SparseFormat;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.engine.MxDeviceType;
import org.apache.mxnet.engine.MxNDArray;

public final class JnaUtils {
    public static final MxnetLibrary LIB = LibUtils.loadLibrary();
    public static final ObjectPool<PointerByReference> REFS =
            new ObjectPool<>(PointerByReference::new, r -> r.setValue(null));

    private static final String[] OP_NAME_PREFIX = {
            "_contrib_", "_linalg_", "_sparse_", "_image_", "_random_"
    };
    private static final Map<String, FunctionInfo> OPS = getNdArrayFunctions();

    public static final String[] EMPTY_ARRAY = new String[0];
    // TODO

    public static void waitToRead(Pointer ndArray) {
        checkNDArray(ndArray, "wait to read");
        checkCall(LIB.MXNDArrayWaitToRead(ndArray));
    }

    public static void waitToWrite(Pointer ndArray) {
        checkNDArray(ndArray, "wait to write");
        checkCall(LIB.MXNDArrayWaitToWrite(ndArray));
    }

    public static Pointer detachGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXNDArrayDetach(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer getGradient(Pointer handle) {
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the gradient for");
        checkCall(LIB.MXNDArrayGetGrad(handle, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static void autogradMarkVariables(
            int numVar, Pointer varHandles, IntBuffer reqsArray, Pointer gradHandles) {
        PointerByReference varRef = REFS.acquire();
        PointerByReference gradRef = REFS.acquire();
        varRef.setValue(varHandles);
        gradRef.setValue(gradHandles);
        checkCall(LIB.MXAutogradMarkVariables(numVar, varRef, reqsArray, gradRef));
        REFS.recycle(varRef);
        REFS.recycle(gradRef);
    }

    public static Map<String, FunctionInfo> getNdArrayFunctions() {
        Set<String> opNames = JnaUtils.getAllOpNames();
        Map<String, FunctionInfo> map = new ConcurrentHashMap<>();

        PointerByReference ref = REFS.acquire();
        for (String opName : opNames) {
            checkCall(LIB.NNGetOpHandle(opName, ref));

            String functionName = getOpNamePrefix(opName);

            // System.out.println("Name: " + opName + "/" + functionName);
            map.put(functionName, getFunctionByName(opName, functionName, ref.getValue()));
            ref.setValue(null);
        }
        REFS.recycle(ref);
        return map;
    }

    public static PairList<Pointer, SparseFormat> imperativeInvoke(
            Pointer function, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        String[] keys;
        String[] values;
        if (params == null) {
            keys = EMPTY_ARRAY;
            values = EMPTY_ARRAY;
        } else {
            keys = params.keyArray(EMPTY_ARRAY);
            values = params.values().stream().map(Object::toString).toArray(String[]::new);
        }
//        StringArray keyArray = StringArray.of(keys);
//        StringArray valueArray = StringArray.of(values);
        PointerArray srcArray = toPointerArray(src);
        PointerArray destArray = toPointerArray(dest);
        PointerByReference destRef = REFS.acquire();
        destRef.setValue(destArray);
        PointerByReference destSType = REFS.acquire();
        IntBuffer numOutputs = IntBuffer.allocate(1);
        numOutputs.put(0, 1);

        checkCall(
                LIB.MXImperativeInvoke(
                        function,
                        src.length,
                        srcArray,
                        numOutputs,
                        destRef,
                        keys.length,
                        keys,
                        values,
                        destSType));
        int numOfOutputs = numOutputs.get(0);
        Pointer[] ptrArray = destRef.getValue().getPointerArray(0, numOfOutputs);
        int[] sTypes = destSType.getValue().getIntArray(0, numOfOutputs);
        PairList<Pointer, SparseFormat> pairList = new PairList<>();
        for (int i = 0; i < numOfOutputs; i++) {
            pairList.add(ptrArray[i], SparseFormat.fromValue(sTypes[i]));
        }
        REFS.recycle(destRef);
        REFS.recycle(destSType);
        srcArray.recycle();
//        keyArray.recycle();
//        valueArray.recycle();

        if (destArray != null) {
            destArray.recycle();
        }
        return pairList;
    }

    private static PointerArray toPointerArray(NDArray[] vals) {
        if (vals == null) {
            return null;
        }
        Pointer[] valPointers = new Pointer[vals.length];
        for (int i = 0; i < vals.length; i++) {
            valPointers[i] = ((MxNDArray) vals[i]).getHandle();
        }
        return PointerArray.of(valPointers);
    }

    public static FunctionInfo op(String opName) {
        if (!OPS.containsKey(opName)) {
            throw new IllegalArgumentException("Unknown operator: " + opName);
        }
        return OPS.get(opName);
    }

    private static FunctionInfo getFunctionByName(
            String name, String functionName, Pointer handle) {
        String[] nameRef = {name};
        String[] description = new String[1];
        IntBuffer numArgs = IntBuffer.allocate(1);
        PointerByReference argNameRef = REFS.acquire();
        PointerByReference argTypeRef = REFS.acquire();
        PointerByReference argDescRef = REFS.acquire();
        String[] keyVarArgs = new String[1];
        String[] returnType = new String[1];

        checkCall(
                LIB.MXSymbolGetAtomicSymbolInfo(
                        handle,
                        nameRef,
                        description,
                        numArgs,
                        argNameRef,
                        argTypeRef,
                        argDescRef,
                        keyVarArgs,
                        returnType));

        int count = numArgs.get();
        PairList<String, String> arguments = new PairList<>();
        if (count != 0) {
            String[] argNames =
                    argNameRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            String[] argTypes =
                    argTypeRef.getValue().getStringArray(0, count, StandardCharsets.UTF_8.name());
            for (int i = 0; i < argNames.length; i++) {
                arguments.add(argNames[i], argTypes[i]);
            }
        }

        REFS.recycle(argNameRef);
        REFS.recycle(argTypeRef);
        REFS.recycle(argDescRef);

        return new FunctionInfo(handle, functionName, arguments);
    }

    public static Set<String> getAllOpNames() {
        IntBuffer outSize = IntBuffer.allocate(1);
        PointerByReference outArray = REFS.acquire();

        checkCall(LIB.MXListAllOpNames(outSize, outArray));

        int size = outSize.get();
        Pointer[] pointers = outArray.getValue().getPointerArray(0, size);

        Set<String> set = new HashSet<>();
        for (Pointer p : pointers) {
            set.add(p.getString(0, StandardCharsets.UTF_8.name()));
        }
        REFS.recycle(outArray);
        return set;
    }

    private static String getOpNamePrefix(String name) {
        for (String prefix : OP_NAME_PREFIX) {
            if (name.startsWith(prefix)) {
                return name.substring(prefix.length());
            }
        }
        return name;
    }


    public static DataType getDataTypeOfNdArray(Pointer handle) {
        IntBuffer dataType = IntBuffer.allocate(1);
        checkNDArray(handle, "get the data type of");
        checkCall(LIB.MXNDArrayGetDType(handle, dataType));
        return DataType.values()[dataType.get()];
    }

    public static Device getDeviceOfNdArray(Pointer handle) {
        IntBuffer deviceType = IntBuffer.allocate(1);
        IntBuffer deviceId = IntBuffer.allocate(1);
        checkNDArray(handle, "get the device of");
        checkCall(LIB.MXNDArrayGetContext(handle, deviceType, deviceId));
        String deviceTypeStr = MxDeviceType.fromDeviceType(deviceType.get(0));
        // CPU is special case which don't have device id
        return Device.of(deviceTypeStr, deviceId.get(0));
    }

    public static Shape getShapeOfNdArray(Pointer handle) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the shape of");
        checkCall(LIB.MXNDArrayGetShape(handle, dim, ref));
        int nDim = dim.get();
        if (nDim == 0) {
            REFS.recycle(ref);
            return new Shape();
        }
        int[] shape = ref.getValue().getIntArray(0, nDim);
        REFS.recycle(ref);
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static Shape getShape64OfNdArray(Pointer handle) {
        IntBuffer dim = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();
        checkNDArray(handle, "get the shape64 of");
        checkCall(LIB.MXNDArrayGetShape64(handle, dim, ref));
        int nDim = dim.get();
        if (nDim == 0) {
            REFS.recycle(ref);
            return new Shape();
        }
        int[] shape = ref.getValue().getIntArray(0, nDim);
        REFS.recycle(ref);
        return new Shape(Arrays.stream(shape).asLongStream().toArray());
    }

    public static SparseFormat getStorageType(Pointer handle) {
        IntBuffer type = IntBuffer.allocate(1);
        checkNDArray(handle, "get the storage type of");
        checkCall(LIB.MXNDArrayGetStorageType(handle, type));
        return SparseFormat.fromValue(type.get());
    }

    public static Pointer createNdArray(
            Device device, Shape shape, DataType dtype, int size, boolean delayedAlloc) {
        int deviceType = MxDeviceType.toDeviceType(device);
        int deviceId = (deviceType != 1) ? device.getDeviceId() : -1;
        int delay = delayedAlloc ? 1 : 0;

        PointerByReference ref = REFS.acquire();
        int[] shapeArray = Arrays.stream(shape.getShape()).mapToInt(Math::toIntExact).toArray();
        checkCall(
                LIB.MXNDArrayCreate(
                        shapeArray, size, deviceType, deviceId, delay, dtype.ordinal(), ref));

        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static int getVersion() {
        IntBuffer version = IntBuffer.allocate(1);
        checkCall(LIB.MXGetVersion(version));
        return version.get();
    }

    public static Pointer createSymbolFromFile(String path) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromFile(path, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    public static Pointer createSymbolFromString(String json) {
        PointerByReference ref = REFS.acquire();
        checkCall(LIB.MXSymbolCreateFromJSON(json, ref));
        Pointer pointer = ref.getValue();
        REFS.recycle(ref);
        return pointer;
    }

    private static String getLastError() {
        return LIB.MXGetLastError();
    }

    private static String[] toStringArray(PointerByReference ref, int size) {
        if (size == 0) {
            return new String[0];
        }

        Pointer[] pointers = ref.getValue().getPointerArray(0, size);

        String[] arr = new String[size];
        for (int i = 0; i < size; ++i) {
            arr[i] = pointers[i].getString(0, StandardCharsets.UTF_8.name());
        }

        return arr;
    }

    public static String[] listSymbolOutputs(Pointer symbol) {
        IntBuffer size = IntBuffer.allocate(1);
        PointerByReference ref = REFS.acquire();

        checkCall(LIB.MXSymbolListOutputs(symbol, size, ref));
        String[] ret = toStringArray(ref, size.get());
        REFS.recycle(ref);
        return ret;
    }

    public static String printSymbol(Pointer symbol) {
        String[] outStr = new String[1];
        checkCall(LIB.NNSymbolPrint(symbol, outStr));
        return outStr[0];
    }

    public static void freeSymbol(Pointer symbol) {
        checkCall(LIB.NNSymbolFree(symbol));
    }

    public static void loadNdArrayFromFile(Path path) {
        loadNdArrayFromFile(path.toAbsolutePath());
    }

    public static PairList<String, Pointer> loadNdArrayFromFile(String path) {
        IntBuffer handleSize = IntBuffer.allocate(1);
        IntBuffer namesSize = IntBuffer.allocate(1);
        PointerByReference handlesRef = REFS.acquire();
        PointerByReference namesRef = REFS.acquire();
        checkCall(LIB.MXNDArrayLoad(path, handleSize, handlesRef, namesSize, namesRef));
        // TODO : construct NDArray Objects
        int handleCount =handleSize.get();
        int nameCount = namesSize.get();
        if (nameCount > 0 && nameCount != handleCount) {
            throw new IllegalStateException(
                    "Mismatch between names and arrays in checkpoint file: " + path);
        }
        Pointer[] handles = handlesRef.getValue().getPointerArray(0, handleCount);

        PairList<String, Pointer> pairList = new PairList<>();

        if (nameCount == 0) {
            for (Pointer handle : handles) {
                pairList.add(null, handle);
            }
        } else {
            String[] names = namesRef.getValue().getStringArray(0, nameCount);
            for (int i = 0; i < handleCount; i++) {
                pairList.add(names[i], handles[i]);
            }
        }
        REFS.recycle(namesRef);
        REFS.recycle(handlesRef);

        return pairList;
    }

    public static void freeNdArray(Pointer handle) {
        checkCall(LIB.MXNDArrayFree(handle));

    }

    public static Pointer loadNdArrayFromByteArray(byte[] buf, int offset, int size) {
        Memory memory = new Memory(size);
        memory.write(0, buf, offset, size);
        PointerByReference outRef = REFS.acquire();
        checkCall(LIB.MXNDArrayLoadFromRawBytes(memory, new NativeSize(size), outRef));
        Pointer p = outRef.getValue();
//        outRef.getValue().getPointerArray(0, size);

        REFS.recycle(outRef);
        return p;
    }

    public static Pointer loadNdArrayFromByteBuffer(ByteBuffer byteBuffer) {
//        Pointer handle = new Pointer(byteBuffer.address);
//        ((DirectByteBuffer) byteBuffer).address()
        // TODO
        byte[] bytes = new byte[byteBuffer.limit()];
        byteBuffer.get(bytes);
        return loadNdArrayFromByteArray(bytes, 0, byteBuffer.limit());
    }

    public static ByteBuffer saveNdArrayAsByteBuffer(Pointer ndArray) {
        NativeSizeByReference size = new NativeSizeByReference();
        PointerByReference ref = new PointerByReference();
        checkCall(LIB.MXNDArraySaveRawBytes(ndArray, size, ref));
        return ref.getValue().getByteBuffer(0, size.getValue().longValue());
    }

    public static byte[] saveNdArrayAsByteArray(Pointer ndArray) {
        ByteBuffer buffer = saveNdArrayAsByteBuffer(ndArray);
        byte[] bytes = new byte[buffer.limit()];
        buffer.get(bytes);
        return bytes;
    }


    public static void syncCopyToCPU(Pointer ndArray, Pointer data, int len) {
        NativeSize size = new NativeSize(len);
        checkNDArray(ndArray, "copy from");
        checkNDArray(data, "copy to");
        checkCall(LIB.MXNDArraySyncCopyToCPU(ndArray, data, size));
    }

    public static void syncCopyFromCPU(Pointer ndArray, Buffer data, int len) {
        NativeSize size = new NativeSize(len);
        Pointer pointer = Native.getDirectBufferPointer(data);
        checkCall(LIB.MXNDArraySyncCopyFromCPU(ndArray, pointer, size));
    }

    private static void checkNDArray(Pointer pointer, String msg) {
        if (pointer == null) {
            throw new IllegalArgumentException(
                    "Tried to " + msg + " an MXNet NDArray that was already closed");
        }
    }

    public static void checkCall(int ret) {
        if (ret != 0) {
            throw new EngineException("MXNet engine call failed: " + getLastError());
        }
    }
}
