package org.apache.mxnet.engine;

import org.apache.mxnet.api.engine.Engine;
import org.apache.mxnet.api.engine.EngineProvider;

/** {@code MxEngineProvider} is the MXNet implementation of {@link EngineProvider}. */
public class MxEngineProvider implements EngineProvider {

    private static volatile Engine engine; // NOPMD

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return MxEngine.ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        if (engine == null) {
            synchronized (this) {
                engine = MxEngine.newInstance();
            }
        }
        return engine;
    }
}