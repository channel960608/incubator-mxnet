package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.engine.Engine;
import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.NDList;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.NDResource;
import org.apache.mxnet.api.ndarray.types.DataType;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.ndarray.types.SparseFormat;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.jna.JnaUtils;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

public class MxNDManager extends BaseNDManager{
    /**
     * A global {@link NDManager} singleton instance.
     *
     * <p>This NDManager is the root of all the other {@code NDManager}s. NDArrays created by this
     * manager are un-managed, so the user has to close them manually. Those NDArrays will be
     * released on GC, and might be run into an out of native memory issue.
     */
    private static final MxNDManager SYSTEM_MANAGER = new SystemManager();

    private static final NDArray[] EMPTY = new NDArray[0];

    private int version;

    private MxNDManager(NDManager parent, Device device, int version) {
        super(parent, device);
        this.version = version;
    }

    static MxNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /**
     * Creates an MxNDArray with the given Native Memory Pointer and attaches to this manager.
     *
     * @param handle the array's native memory pointer
     * @return the created array
     */
    public MxNDArray create(Pointer handle) {
        if (version >= 10700) {
            return new MxNDArray(this, handle);
        }
        // TODO
        return null;
    }

    /**
     * Creates a sparse MxNDArray with the given Native Memory Pointer and attaches to this manager.
     *
     * @param handle the array's native memory pointer
     * @param fmt the sparse format to use
     * @return the created array
     */
    public MxNDArray create(Pointer handle, SparseFormat fmt) {
        return new MxNDArray(this, handle, fmt);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray create(Shape shape, DataType dataType) {
        Pointer handle = JnaUtils.createNdArray(device, shape, dataType, shape.dimension(), false);
        if (version >= 10700) {
            return new MxNDArray(this, handle, device, shape, dataType, false);
        }
        // TODO: MxNDArray16
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        SparseFormat fmt = SparseFormat.CSR;
        DataType dataType = DataType.fromBuffer(data);
        MxNDArray indptrNd = create(new Shape(indptr.length), DataType.INT64);
        indptrNd.set(indptr);
        MxNDArray indicesNd = create(new Shape(indices.length), DataType.INT64);
        indicesNd.set(indices);
        Pointer handle =
                JnaUtils.createSparseNdArray(
                        fmt,
                        device,
                        shape,
                        dataType,
                        new DataType[] {indptrNd.getDataType(), indicesNd.getDataType()},
                        new Shape[] {indptrNd.getShape(), indicesNd.getShape()},
                        false);
        MxNDArray sparse = create(handle, fmt);
        MxNDArray dataNd = create(new Shape(data.remaining()), dataType);
        dataNd.set(data);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, dataNd, -1);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indptrNd, 0);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indicesNd, 1);
        return sparse;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        SparseFormat fmt = SparseFormat.ROW_SPARSE;
        DataType dataType = DataType.fromBuffer(data);
        MxNDArray indicesNd = create(new Shape(indices.length), DataType.INT64);
        indicesNd.set(indices);
        Pointer handle =
                JnaUtils.createSparseNdArray(
                        fmt,
                        device,
                        shape,
                        dataType,
                        new DataType[] {indicesNd.getDataType()},
                        new Shape[] {indicesNd.getShape()},
                        false);
        MxNDArray sparse = create(handle, fmt);
        MxNDArray dataNd = create(dataShape, dataType);
        dataNd.set(data);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, dataNd, -1);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indicesNd, 0);
        return sparse;
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        return JnaUtils.loadNdArray(this, path, device);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        return fill("_npi_zeros", shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        return fill("_npi_ones", shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("shape", shape);
        params.addParam("value", value);
        params.setDataType(dataType);
        params.setDevice(device);
        return invoke("_npi_full", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("step", step);
        params.setDataType(dataType);
        params.setDevice(device);
        return invoke("_npi_arange", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("N", rows);
        params.addParam("M", cols);
        params.addParam("k", k);
        params.setDataType(dataType);
        params.setDevice(device);
        return invoke("_npi_eye", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        if (num < 0) {
            throw new IllegalArgumentException("Num argument must be non-negative");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("num", num);
        params.addParam("endpoint", endpoint);
        params.setDevice(device);
        return invoke("_npi_linspace", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("low", low);
        params.addParam("high", high);
        params.addParam("shape", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke("_npi_random_randint", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("low", low);
        params.addParam("high", high);
        params.addParam("size", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke("_npi_uniform", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        params.addParam("loc", loc);
        params.addParam("scale", scale);
        params.addParam("size", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke("_npi_normal", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        MxOpParams params = new MxOpParams();
        params.addParam("n", n);
        params.addParam("size", shape);
        return invoke("_npi_multinomial", pValues, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        MxOpParams params = new MxOpParams();
        params.addParam("n", n);
        return invoke("_npi_multinomial", pValues, params);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDManager newSubManager(Device dev) {
        MxNDManager manager = new MxNDManager(this, dev, version);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        JnaUtils.op(operation).invoke(src, dest, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return new NDList(JnaUtils.op(operation).invoke(this, src.toArray(EMPTY), params));
    }

    @Override
    public Engine getEngine() {
        return Engine.getEngine();
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {
        invoke(operation, src.toArray(EMPTY), dest.toArray(EMPTY), params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the array of source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray[] src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(this, src, params)[0];
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray src, PairList<String, ?> params) {
        return invoke(operation, new NDArray[] {src}, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, PairList<String, ?> params) {
        return invoke(operation, EMPTY, params);
    }

    private NDArray fill(String opName, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        if (shape == null) {
            throw new IllegalArgumentException("Shape is required for " + opName.substring(1));
        }
        params.addParam("shape", shape);
        params.setDevice(device);
        params.setDataType(dataType);
        return invoke(opName, params);
    }

    /** The SystemManager is the root {@link MxNDManager} of which all others are children. */
    private static final class SystemManager extends MxNDManager {

        SystemManager() {
            super(null, null, JnaUtils.getVersion());
        }

        /** {@inheritDoc} */
        @Override
        public void attachInternal(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void tempAttachInternal(
                NDManager originalManager, String resourceId, NDResource resource) {}

        /** {@inheritDoc} */
        @Override
        public void detachInternal(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }

}
