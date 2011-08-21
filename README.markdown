                                         _ __  
                                        | '_ \ 
                                        | | | |
      /                             \   |_| |_| 
     /                               \        
     |      +             /          |        
     |       +     +     /           |      
     |                  /   -        |      ____  _   _         _  _  _      _  _                            
     |     +    +   +  /  -          |     / ___|| | (_)       | |(_)| |__  | |(_) _ __    ___   __ _  _ __  
     |                /        -     |    | |    | | | | _____ | || || '_ \ | || || '_ \  / _ \ / _` || '__| 
     |     +   +     / -   -         |    | |___ | | | ||_____|| || || |_) || || || | | ||  __/| (_| || |    
     |              /                |     \____||_|_/ |       |_||_||_.__/ |_||_||_| |_| \___| \__,_||_|    
     |    +        /   -    -        |             |__/                                                      
     |          + /      -           |    
     \           /   -               /    
      \                             /                                                           


This is a Clojure wrapper around Benedikt Waldvogel's [Java port](http://www.bwaldvogel.de/liblinear-java) of [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), a linear classifier that can handle problems with millions of instances and features.
Essentially, it is a support vector machine optimized for classes that can be separated without projecting into some fancy-pants kernel space.

```clojure
(use '[clj-liblinear.core :only [train predict]])
(let [train-data (concat
                  (repeatedly 300 #(hash-map :class 0 :f {:x (rand), :y (rand)}))
                  (repeatedly 300 #(hash-map :class 1 :f {:x (- (rand)), :y (- (rand))})))
      model (train
             (map :f train-data)
             (map :class train-data)
             :algorithm :l2l2)]
  
  [(predict model {:x (rand) :y (rand)})
   (predict model {:x (- (rand)) :y (- (rand))})])
;;=> [0 1]
```



Install
=======

Add

```clojure
[clj-liblinear "0.0.1-SNAPSHOT"]
```

to the `:dependencies` vector in your `projects.clj` file, then run `cake deps` or `lein deps`.

If you want to be on the bleeding edge, add this git repository as a submodule on your project:

```bash
mkdir vendor
git submodule add https://github.com/lynaghk/clj-liblinear vendor/clj-liblinear
```

and then setup the classpath appropriately.
If you are using `cake`, for instance, open up `<your repo>/.cake/config` and add the line

    project.classpath = vendor/clj-liblinear/src




Thanks
======
The National Taiwan University Machine Learning Group for [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/), and Benedikt Waldvogel his [Java transliteration](http://www.bwaldvogel.de/liblinear-java).

This project is sponsored by [Keming Labs](http://keminglabs.com), a technical design studio specializing in data visualization.
