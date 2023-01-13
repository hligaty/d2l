package org.example;

import ai.djl.ndarray.NDArray;

public abstract class BaseTest {
    void println(String s) {
        System.out.println(s);
    }

    void println(String desc, NDArray ndArray) {
        System.out.println((desc == null ? "" : (desc + ": ")) + ndArray.toDebugString(true));
    }
}
