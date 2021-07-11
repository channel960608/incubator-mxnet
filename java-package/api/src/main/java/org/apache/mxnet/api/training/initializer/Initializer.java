package org.apache.mxnet.api.training.initializer;

import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.types.DataType;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.nn.Block;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link NDArray} parameters stored within a {@link Block}.
 *
 * @see <a
 *     href="https://d2l.djl.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html">The
 *     D2L chapter on numerical stability and initialization</a>
 */
public interface Initializer {

    Initializer ZEROS = (m, s, t) -> m.zeros(s, t, m.getDevice());
    Initializer ONES = (m, s, t) -> m.ones(s, t, m.getDevice());

    /**
     * Initializes a single {@link NDArray}.
     *
     * @param shape the {@link Shape} for the new NDArray
     * @param dataType the {@link DataType} for the new NDArray
     * @return the {@link NDArray} initialized with the manager and shape
     */
    NDArray initialize(NDManager manager, Shape shape, DataType dataType);
}