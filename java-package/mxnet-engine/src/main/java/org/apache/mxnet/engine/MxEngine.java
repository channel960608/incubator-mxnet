package org.apache.mxnet.engine;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.engine.Engine;
import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.nn.SymbolBlock;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class MxEngine extends Engine {

    private static final Logger logger = LoggerFactory.getLogger(MxEngine.class);
    public static final String ENGINE_NAME = "MXNet";
    static final int RANK = 1;
    private static final String MXNET_EXTRA_LIBRARY_VERBOSE = "MXNET_EXTRA_LIBRARY_VERBOSE";

    static Engine newInstance() {
        try {
            // Workaround MXNet engine lazy initialization issue
            JnaUtils.getAllOpNames();

            JnaUtils.setNumpyMode(JnaUtils.NumpyMode.GLOBAL_ON);

            // Workaround MXNet shutdown crash issue
            Runtime.getRuntime().addShutdownHook(new Thread(JnaUtils::waitAll)); // NOPMD

            // load extra MXNet library
            String paths = System.getenv("MXNET_EXTRA_LIBRARY_PATH");
            boolean extraLibVerbose = false;
            if (System.getenv().containsKey(MXNET_EXTRA_LIBRARY_VERBOSE)) {
                extraLibVerbose = Boolean.parseBoolean(System.getenv(MXNET_EXTRA_LIBRARY_VERBOSE));
            }
            if (paths != null) {
                String[] files = paths.split(",");
                for (String file : files) {
                    Path path = Paths.get(file);
                    if (Files.notExists(path)) {
                        throw new FileNotFoundException("Extra Library not found: " + file);
                    }
                    JnaUtils.loadLib(path.toAbsolutePath().toString(), extraLibVerbose);
                }
            }

            return new MxEngine();
        } catch (Throwable t) {
            throw new EngineException("Failed to load MXNet native library", t);
        }
    }

//    private static synchronized MxEngineProvider initEngineProvider() {
//        ServiceLoader<MxEngineProvider> loaders = ServiceLoader.load(MxEngineProvider.class);
//        Iterator<MxEngineProvider> loaderIterator = loaders.iterator();
//        if (loaderIterator.hasNext()) {
//            MxEngineProvider engineProvider = loaderIterator.next();
//            logger.debug("Found EngineProvider for engine: {}", engineProvider.getEngineName());
//            return engineProvider;
//        } else {
//            logger.debug("No EngineProvider found");
//            return null;
//        }
//    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public boolean hasCapability(String capability) {
        return JnaUtils.getFeatures().contains(capability);
    }

    @Override
    public NDManager newBaseManager() {
        return null;
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return new MxSymbolBlock(manager);
    }
}
