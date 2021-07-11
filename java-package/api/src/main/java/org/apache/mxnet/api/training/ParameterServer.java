package org.apache.mxnet.api.training;

import org.apache.mxnet.api.ndarray.NDArray;

import java.util.Arrays;

/** An interface for a key-value store to store parameters, and their corresponding gradients. */
public interface ParameterServer extends AutoCloseable {

    /**
     * Initializes the {@code ParameterStore} for the given parameter.
     *
     * @param parameterId the parameter ID
     * @param value the values to be set for the given parameter
     */
    void init(String parameterId, NDArray[] value);

    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param params the parameter NDArrays in different devices to be updated.
     */
    default void update(String parameterId, NDArray[] params) {
        NDArray[] grads = Arrays.stream(params).map(NDArray::getGradient).toArray(NDArray[]::new);
        update(parameterId, grads, params);
        Arrays.stream(grads).forEach(NDArray::close);
    }
    /**
     * Updates the parameter of a key from Parameter Server.
     *
     * @param parameterId the key to identify the parameter
     * @param grads the gradient NDArrays in different devices to apply the update.
     * @param params the parameter NDArrays in different devices to be updated.
     */
    void update(String parameterId, NDArray[] grads, NDArray[] params);

    /** {@inheritDoc} */
    @Override
    void close();
}