package org.apache.mxnet.engine;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.engine.Engine;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.nn.SymbolBlock;

public class MxEngine extends Engine {

    public static final String ENGINE_NAME = "MXNet";

    @Override
    public String getEngineName() {
        return null;
    }

    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    @Override
    public NDManager newBaseManager() {
        return null;
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return null;
    }

    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return null;
    }
}
