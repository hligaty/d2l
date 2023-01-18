package org.example;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;

public abstract class BaseTest {
    static NDManager manager;
    
    @BeforeAll
    public static void before() {
        manager = NDManager.newBaseManager(Device.cpu(), "MXNet");
    }

    @AfterAll
    public static void after() {
        manager.close();
    }
    
    static class Tuple<A, B> {
        A a;
        B b;

        public Tuple(A a, B b) {
            this.a = a;
            this.b = b;
        }
    }
    
    static void println(String s) {
        System.out.println(s);
    }

    static void println(String desc, NDArray ndArray) {
        System.out.println((desc == null ? "" : (desc + ": ")) + ndArray.toDebugString(true));
    }
}
