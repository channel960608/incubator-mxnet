package org.apache.mxnet.api.nn;

import org.apache.mxnet.api.util.Pair;
import org.apache.mxnet.api.util.PairList;

import java.util.List;
import java.util.Map;

/** Represents a set of names and Blocks. */
public class BlockList extends PairList<String, Block> {

    /** Creates an empty {@code BlockList}. */
    public BlockList() {}

    /**
     * Constructs an empty {@code BlockList} with the specified initial capacity.
     *
     * @param initialCapacity the initial capacity of the list
     * @throws IllegalArgumentException if the specified initial capacity is negative
     */
    public BlockList(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified keys and values.
     *
     * @param keys the key list containing the elements to be placed into this {@code BlockList}
     * @param values the value list containing the elements to be placed into this {@code BlockList}
     * @throws IllegalArgumentException if the keys and values size are different
     */
    public BlockList(List<String> keys, List<Block> values) {
        super(keys, values);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified list of Pairs.
     *
     * @param list the list containing the elements to be placed into this {@code BlockList}
     */
    public BlockList(List<Pair<String, Block>> list) {
        super(list);
    }

    /**
     * Constructs a {@code BlockList} containing the elements of the specified map.
     *
     * @param map the map containing keys and values
     */
    public BlockList(Map<String, Block> map) {
        super(map);
    }
}
