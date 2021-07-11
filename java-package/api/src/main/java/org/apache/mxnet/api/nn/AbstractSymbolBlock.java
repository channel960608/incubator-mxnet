package org.apache.mxnet.api.nn;

import org.apache.mxnet.api.ndarray.types.Shape;

/** {@code AbstractSymbolBlock} is an abstract implementation of {@link SymbolBlock}. */
public abstract class AbstractSymbolBlock extends AbstractBlock implements SymbolBlock {

    /**
     * Builds an empty block with the given version for parameter serialization.
     *
     * @param version the version to use for parameter serialization.
     */
    public AbstractSymbolBlock(byte version) {
        super(version);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("not implement!");
    }
}