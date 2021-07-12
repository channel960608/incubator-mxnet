package org.apache.mxnet.engine;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.api.util.Utils;
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
    private MxNDManager manager;

    /**
     * Constructs a {@code Symbol}.
     *
     * @param manager the manager to attach the symbol to
     * @param pointer the symbol's native data location
     */
    Symbol(MxNDManager manager, Pointer pointer) {
        super(pointer);
        this.manager = manager;
//        manager.attachInternal(getUid(), this);
        //        argParams = JnaUtils.listSymbolArguments(getHandle());
        //        auxParams = JnaUtils.listSymbolAuxiliaryStates(getHandle());
    }


    public static Symbol loadFromFile(MxNDManager manager,String path) {
        Pointer p = JnaUtils.createSymbolFromFile(path);
        return new Symbol(manager, p);
    }

    public static Symbol load(MxNDManager manager, String path) {
        Pointer pointer = JnaUtils.createSymbolFromFile(path);
        return new Symbol(manager, pointer);
    }

    /**
     * Loads a symbol from a json string.
     *
     * @param json the json string of the symbol.
     * @return the new symbol
     */
    public static Symbol loadJson(MxNDManager manager, String json) {
        Pointer pointer = JnaUtils.createSymbolFromString(json);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the output symbol by index.
     *
     * @param index the index of the output
     * @return the symbol output as a new symbol
     */
    public Symbol get(int index) {
        Pointer pointer = JnaUtils.getSymbolOutput(getInternals().getHandle(), index);
        return new Symbol(manager, pointer);
    }

    /**
     * Returns the output symbol with the given name.
     *
     * @param name the name of the symbol to return
     * @return the output symbol
     * @throws IllegalArgumentException Thrown if no output matches the name
     */
    public Symbol get(String name) {
        String[] out = getInternalOutputNames();
        int index = Utils.indexOf(out, name);
        if (index < 0) {
            throw new IllegalArgumentException("Cannot find output that matches name: " + name);
        }
        return get(index);
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
        return new Symbol(manager, pointer);
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
        return new Symbol(manager, JnaUtils.optimizeFor(this, backend, device));
    }

    /**
     * Converts Symbol to json string for saving purpose.
     *
     * @return the json string
     */
    public String toJsonString() {
        return JnaUtils.getSymbolString(getHandle());
    }

}
