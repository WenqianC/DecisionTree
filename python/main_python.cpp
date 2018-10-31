//
//  main.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-07-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#if 0
#include <iostream>

class Foo {
public:
    void bar() {
        std::cout<<"Hello "<<std::endl;
    }
};

extern "C" {
    Foo* Foo_new() {
        return new Foo();
    }
    void Foo_bar(Foo* foo) {
        foo->bar();
    }
}

#endif

