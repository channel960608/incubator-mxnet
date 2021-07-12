package org.apache.mxnet.engine;

import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.NDList;
import org.apache.mxnet.api.ndarray.dim.full.NDIndexFullPick;
import org.apache.mxnet.api.ndarray.dim.full.NDIndexFullSlice;
import org.apache.mxnet.api.ndarray.index.NDArrayIndexer;
import org.apache.mxnet.api.ndarray.types.Shape;

import java.util.Stack;

/** The {@link NDArrayIndexer} used by the {@link MxNDArray}. */
public class MxNDArrayIndexer extends NDArrayIndexer {

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullPick fullPick) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", fullPick.getAxis());
        params.addParam("keepdims", true);
        params.add("mode", "wrap");
        return array.getManager()
                .invoke("pick", new NDList(array, fullPick.getIndices()), params)
                .singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        NDArray result = ((MxNDManager) array.getManager()).invoke("_npi_slice", array, params);
        int[] toSqueeze = fullSlice.getToSqueeze();
        if (toSqueeze.length > 0) {
            NDArray oldResult = result;
            result = result.squeeze(toSqueeze);
            oldResult.close();
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        Stack<NDArray> prepareValue = new Stack<>();
        prepareValue.add(value);
        prepareValue.add(prepareValue.peek().toDevice(array.getDevice(), false));
        // prepareValue.add(prepareValue.peek().asType(getDataType(), false));
        // Deal with the case target: (1, 10, 1), original (10)
        // try to find (10, 1) and reshape (10) to that
        Shape targetShape = fullSlice.getShape();
        while (targetShape.size() > value.size()) {
            targetShape = targetShape.slice(1);
        }
        prepareValue.add(prepareValue.peek().reshape(targetShape));
        prepareValue.add(prepareValue.peek().broadcast(fullSlice.getShape()));

        array.getManager()
                .invoke(
                        "_npi_slice_assign",
                        new NDArray[] {array, prepareValue.peek()},
                        new NDArray[] {array},
                        params);
        for (NDArray toClean : prepareValue) {
            if (toClean != value) {
                toClean.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());
        params.addParam("scalar", value);
        array.getManager()
                .invoke(
                        "_npi_slice_assign_scalar",
                        new NDArray[] {array},
                        new NDArray[] {array},
                        params);
    }
}