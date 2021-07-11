package org.apache.mxnet.engine;

import org.apache.mxnet.api.exception.MalformedModelException;
import org.apache.mxnet.api.ndarray.NDList;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.nn.AbstractSymbolBlock;
import org.apache.mxnet.api.nn.Parameter;
import org.apache.mxnet.api.training.ParameterStore;
import org.apache.mxnet.api.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MxSymbolBlock extends AbstractSymbolBlock {

    private static final Logger logger = LoggerFactory.getLogger(MxSymbolBlock.class);
    private static final byte VERSION = 3;

    private CachedOp op;
    private Symbol symbol;
    private List<Parameter> mxNetParams; // includes input data
    private Map<String, Shape> paramShapes;
    private Shape[] outputShapes;
    private PairList<String, Shape> inputDescriptions;
    private PairList<String, Shape> outputDescriptions;
    private boolean first;

    /**
     * Constructs a {@code MxSymbolBlock} for a {@link Symbol}.
     *
     * <p>You can create a {@code MxSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param symbol the symbol containing the block's symbolic graph
     */
    public MxSymbolBlock(Symbol symbol) {
        super(VERSION);
        this.symbol = symbol;
        initBlock();
    }

    /**
     * Constructs an empty {@code MxSymbolBlock}.
     *
     */
    public MxSymbolBlock() {
        super(VERSION);
    }

    /**
     * Sets the names of the input data.
     *
     * @param inputNames the names of the input data
     */
    public void setInputNames(List<String> inputNames) {
        this.inputNames = inputNames;
        // now that we know which of the parameters are just input placeholders and which
        // are trainable, add them properly so they are correctly handled
        Set<String> nameLookup = new HashSet<>(inputNames);
        for (Parameter mxNetParameter : mxNetParams) {
            if (!nameLookup.contains(mxNetParameter.getName())) {
                addParameter(mxNetParameter);
            }
        }
    }

    /**
     * Returns the list of inputs and parameter NDArrays.
     *
     * @return the list of inputs and parameter NDArrays
     */
    public List<Parameter> getAllParameters() {
        return mxNetParams;
    }

    /**
     * Returns the layers' name.
     *
     * @return a List of String containing the layers' name
     */
    public List<String> getLayerNames() {
        return symbol.getLayerNames();
    }

    /**
     * Returns the Symbolic graph from the model.
     *
     * @return a {@link Symbol} object
     */
    public Symbol getSymbol() {
        return symbol;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return null;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException, MalformedModelException {

    }

    private void initBlock() {
        inputNames = new ArrayList<>();

        String[] allNames = symbol.getAllNames();
        mxNetParams = new ArrayList<>(allNames.length);

        Set<String> auxNameSet = new HashSet<>(Arrays.asList(symbol.getAuxNames()));
        for (String name : allNames) {
            Parameter.Type type = inferType(name);
            boolean requireGrad = !auxNameSet.contains(name);
            mxNetParams.add(
                    Parameter.builder()
                            .setName(name)
                            .setType(type)
                            .optRequiresGrad(requireGrad)
                            .build());
        }
        first = true;
    }

    private static Parameter.Type inferType(String name) {
        if (name.endsWith("bias")) {
            return Parameter.Type.BIAS;
        } else if (name.endsWith("gamma")) {
            return Parameter.Type.GAMMA;
        } else if (name.endsWith("beta")) {
            return Parameter.Type.BETA;
        } else if (name.endsWith("moving_mean") || name.endsWith("running_mean")) {
            return Parameter.Type.RUNNING_MEAN;
        } else if (name.endsWith("moving_var") || name.endsWith("running_var")) {
            return Parameter.Type.RUNNING_VAR;
        } else if (name.endsWith("weight")) {
            return Parameter.Type.WEIGHT;
        }
        return Parameter.Type.OTHER;
    }
}
