package org.apache.mxnet.engine;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.NDList;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.NDResource;
import org.apache.mxnet.api.ndarray.types.DataType;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.Buffer;
import java.nio.file.Path;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/** {@code BaseNDManager} is the default implementation of {@link NDManager}. */
public abstract class BaseNDManager implements NDManager {

    private static final Logger logger = LoggerFactory.getLogger(BaseNDManager.class);

    protected NDManager parent;
    protected String uid;
    protected String name;
    protected Device device;
    protected ConcurrentHashMap<String, AutoCloseable> resources;
    protected ConcurrentHashMap<String, TempResource> tempResources;
    protected AtomicBoolean closed = new AtomicBoolean(false);

    protected BaseNDManager(NDManager parent, Device device) {
        this.parent = parent;
//        this.device = Device.defaultIfNull(device, getEngine());
        this.device = Device.DEFAULT_DEVICE;
        resources = new ConcurrentHashMap<>();
        tempResources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String data) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCoo(Buffer data, long[][] indices, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return this.name == null ? uid : this.name;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isOpen() {
        return !closed.get();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        String parentName = parent == null ? "No Parent" : parent.getName();
        return "Name: "
                + getName()
                + " Parent Name: "
                + parentName
                + " isOpen: "
                + isOpen()
                + " Resource size: "
                + resources.size();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void attachInternal(String resourceId, AutoCloseable resource) {
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        tempResources.compute(
                resourceId,
                (key, tempResource) -> {
                    if (tempResource != null) {
                        // This state occurs when this manager (manA) tempAttaches a resource that
                        // is later
                        // tempAttached to another manager (manB)
                        // When manB is closed, it will use attach to return the resource to this
                        // (manA)
                        // In that case, it should stay as a tempResource in this (manA)
                        tempResource.detached = false;
                    } else {
                        resources.put(resourceId, resource);
                    }
                    return tempResource;
                });
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttachInternal(
            NDManager originalManager, String resourceId, NDResource resource) {
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        tempResources.put(resourceId, new TempResource(resource, originalManager));
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void detachInternal(String resourceId) {
        if (closed.get()) {
            // This may happen in the middle of BaseNDManager.close()
            return;
        }
        tempResources.computeIfPresent(
                resourceId,
                (key, tempResource) -> {
                    tempResource.detached = true;
                    return tempResource;
                });
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not supported!");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (!closed.getAndSet(true)) {
            for (AutoCloseable closeable : resources.values()) {
                try {
                    closeable.close();
                } catch (Exception e) {
                    logger.error("Resource close failed.", e);
                }
            }
            for (TempResource resource : tempResources.values()) {
                resource.returnResource();
            }
            parent.detachInternal(uid);
            resources.clear();
            tempResources.clear();
        }
    }

    /**
     * Prints information about this {@link NDManager} and all sub-managers to the console.
     *
     * @param level the level of this {@link NDManager} in the hierarchy
     */
    public void debugDump(int level) {
        StringBuilder sb = new StringBuilder(100);
        for (int i = 0; i < level; ++i) {
            sb.append("    ");
        }
        sb.append("\\--- NDManager(")
                .append(uid.substring(24))
                .append(") resource count: ")
                .append(resources.size());

        System.out.println(sb); // NOPMD
        for (AutoCloseable c : resources.values()) {
            if (c instanceof BaseNDManager) {
                ((BaseNDManager) c).debugDump(level + 1);
            }
        }
    }

    protected static final class TempResource {

        private NDResource resource;
        private NDManager manager;
        private boolean detached;

        public TempResource(NDResource resource, NDManager manager) {
            this.resource = resource;
            this.manager = manager;
            this.detached = false;
        }

        public void returnResource() {
            try {
                if (!detached) {
                    if (manager.isOpen()) {
                        resource.attach(manager);
                    } else {
                        resource.close();
                    }
                }
            } catch (Exception e) {
                logger.error("Temporary resource return failed.", e);
            }
        }
    }
}