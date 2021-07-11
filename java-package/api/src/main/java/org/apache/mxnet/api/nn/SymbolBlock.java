package org.apache.mxnet.api.nn;

import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.util.PairList;

/**
 * {@code SymbolBlock} is a {@link Block} is used to load models that were exported directly from
 * the engine in its native format.
 */
public interface SymbolBlock extends Block {

    /**
     * Creates an empty SymbolBlock instance.
     *
     * @param manager the manager to be applied in the SymbolBlock
     * @return a new Model instance
     */
    static SymbolBlock newInstance(NDManager manager) {
        return manager.getEngine().newSymbolBlock(manager);
    }

    /** Removes the last block in the symbolic graph. */
    default void removeLastBlock() {
        throw new UnsupportedOperationException("not supported");
    }

    /**
     * Returns a {@link PairList} of output names and shapes stored in model file.
     *
     * @return the {@link PairList} of output names, and shapes
     */
    default PairList<String, Shape> describeOutput() {
        throw new UnsupportedOperationException("not supported");
    }
}