package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.api.util.NativeResource;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

// TODO
public class Symbol extends NativeResource<Pointer> {

    private String[] outputs;

    protected Symbol(Pointer handle) {
        super(handle);
    }

    public static Symbol loadFromFile(String path) {
        Pointer p = JnaUtils.createSymbolFromFile(path);
        return new Symbol(p);
    }

    public static Symbol load(String path) {
        Pointer pointer = JnaUtils.createSymbolFromFile(path);
        return new Symbol(pointer);
    }

    /**
     * Loads a symbol from a json string.
     *
     * @param json the json string of the symbol.
     * @return the new symbol
     */
    public static Symbol loadJson(String json) {
        Pointer pointer = JnaUtils.createSymbolFromString(json);
        return new Symbol(pointer);
    }

    /**
     * Returns the symbol argument names.
     *
     * @return the symbol argument names
     */
    public String[] getArgNames() {
        return JnaUtils.listSymbolArguments(getHandle());
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Pointer pointer = handle.getAndSet(null);
        if (pointer != null) {
            JnaUtils.freeSymbol(pointer);
        }
    }

    /**
     * Returns the MXNet auxiliary states for the symbol.
     *
     * @return the MXNet auxiliary states for the symbol
     */
    public String[] getAuxNames() {
        return JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }

    /**
     * Returns the symbol names.
     *
     * @return the symbol names
     */
    public String[] getAllNames() {
        return JnaUtils.listSymbolNames(getHandle());
    }

    /**
     * Returns the symbol outputs.
     *
     * @return the symbol outputs
     */
    public String[] getOutputNames() {
        if (outputs == null) {
            outputs = JnaUtils.listSymbolOutputs(getHandle());
        }
        return outputs;
    }

    /**
     * Returns the list of names for all internal outputs.
     *
     * @return a list of names
     */
    public List<String> getLayerNames() {
        String[] outputNames = getInternalOutputNames();
        String[] allNames = getAllNames();
        Set<String> allNamesSet = new LinkedHashSet<>(Arrays.asList(allNames));
        // Kill all params field and keep the output layer
        return Arrays.stream(outputNames)
                .filter(n -> !allNamesSet.contains(n))
                .collect(Collectors.toList());
    }

    private String[] getInternalOutputNames() {
        return JnaUtils.listSymbolOutputs(getInternals().getHandle());
    }

    /**
     * Returns the symbol internals.
     *
     * @return the symbol internals symbol
     */
    public Symbol getInternals() {
        Pointer pointer = JnaUtils.getSymbolInternals(getHandle());
        return new Symbol(pointer);
    }

    /**
     * Infers the shapes for all parameters inside a symbol from the given input shapes.
     *
     * @param pairs the given input name and shape
     * @return a map of arguments with names and shapes
     */
    public Map<String, Shape> inferShape(PairList<String, Shape> pairs) {
        List<List<Shape>> shapes = JnaUtils.inferShape(this, pairs);
        if (shapes == null) {
            throw new IllegalArgumentException("Cannot infer shape based on the data provided!");
        }
        List<Shape> argShapes = shapes.get(0);
        List<Shape> outputShapes = shapes.get(1);
        List<Shape> auxShapes = shapes.get(2);
        // TODO: add output to the map
        String[] argNames = getArgNames();
        String[] auxNames = getAuxNames();
        String[] outputNames = getOutputNames();
        Map<String, Shape> shapesMap = new ConcurrentHashMap<>();
        for (int i = 0; i < argNames.length; i++) {
            shapesMap.put(argNames[i], argShapes.get(i));
        }
        for (int i = 0; i < auxNames.length; i++) {
            shapesMap.put(auxNames[i], auxShapes.get(i));
        }
        for (int i = 0; i < outputNames.length; i++) {
            shapesMap.put(outputNames[i], outputShapes.get(i));
        }
        return shapesMap;
    }

    /**
     * [Experimental] Add customized optimization on the Symbol.
     *
     * <p>This method can be used with EIA or TensorRT for model acceleration
     *
     * @param backend backend name
     * @param device the device assigned
     * @return optimized Symbol
     */
    public Symbol optimizeFor(String backend, Device device) {
        return new Symbol(JnaUtils.optimizeFor(this, backend, device));
    }

}
