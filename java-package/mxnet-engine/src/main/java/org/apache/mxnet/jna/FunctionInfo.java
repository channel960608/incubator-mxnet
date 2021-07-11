package org.apache.mxnet.jna;

import com.sun.jna.Pointer;
import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.types.SparseFormat;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.engine.MxNDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A FunctionInfo represents an operator (ie function) within the MXNet Engine. */
public class FunctionInfo {

    private Pointer handle;
    private String name;
    private PairList<String, String> arguments;

    private static final Logger logger = LoggerFactory.getLogger(FunctionInfo.class);

    public String getName() {
        return this.name;
    }

    FunctionInfo(Pointer pointer, String functionName, PairList<String, String> arguments) {
        this.handle = pointer;
        this.name = functionName;
        this.arguments = arguments;
    }

    // TODO : complete other methods
    /**
     * Calls an operator with the given arguments.
     *
     * @param src the input NDArray(s) to the operator
     * @param dest the destination NDArray(s) to be overwritten with the result of the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     * @return the error code or zero for no errors
     */
    public int invoke(
            NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        checkDevices(src);
        checkDevices(dest);
        return JnaUtils.imperativeInvoke(handle, src, dest, params).size();
    }

    /**
     * Calls an operator with the given arguments.
     *
     * @param src the input NDArray(s) to the operator
     * @param params the non-NDArray arguments to the operator. Should be a {@code PairList<String,
     *     String>}
     * @return the error code or zero for no errors
     */
    public NDArray[] invoke(NDArray[] src, PairList<String, ?> params) {
        checkDevices(src);
        PairList<Pointer, SparseFormat> pairList =
                JnaUtils.imperativeInvoke(handle, src, null, params);
        return pairList.stream()
                .map(
                        pair -> {
                            if (pair.getValue() != SparseFormat.DENSE) {
                                return new MxNDArray(pair.getKey(), pair.getValue());
                            }
                            return new MxNDArray(pair.getKey());
                        })
                .toArray(MxNDArray[]::new);
    }

    private void checkDevices(NDArray[] src) {
        // check if all the NDArrays are in the same device
        if (logger.isDebugEnabled() && src.length > 1) {
            Device device = src[0].getDevice();
            for (int i = 1; i < src.length; ++i) {
                if (!device.equals(src[i].getDevice())) {
                    logger.warn(
                            "Please make sure all the NDArrays are in the same device. You can call toDevice() to move the NDArray to the desired Device.");
                }
            }
        }
    }
}
