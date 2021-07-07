package org.apache.mxnet.api.ndarray.dim;

import org.apache.mxnet.api.ndarray.NDArray;

/** An {@code NDIndexElement} to return values based on a mask binary NDArray. */
public class NDIndexBooleans implements NDIndexElement {

    private NDArray index;

    /**
     * Constructs a {@code NDIndexBooleans} instance with specified mask binary NDArray.
     *
     * @param index the mask binary {@code NDArray}
     */
    public NDIndexBooleans(NDArray index) {
        this.index = index;
    }

    /**
     * Returns the mask binary {@code NDArray}.
     *
     * @return the mask binary {@code NDArray}
     */
    public NDArray getIndex() {
        return index;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return index.getShape().dimension();
    }
}