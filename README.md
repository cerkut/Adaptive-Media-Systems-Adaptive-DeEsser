# Project/Product
Describe

# Building

## Windows

TODO

## MacOS

In the root of the repo issue 

``` shell
../JUCE/Projucer.app/Contents/MacOS/Projucer --resave TestDeEs.jucer
```
where the first path is where your `Projucer` resides. This will create the .xcodeproject under `Builds/MacOSX/`. Then, to compile, issue


``` shell
xcodebuild -project Builds/MacOSX/TestDeEs.xcodeproj
```

You can then check if the standalone app works

``` shell
open Builds/MacOSX/build/Debug/TestDeEs.app
```

## cmake builds

Requires cmake installed. In the root of the repo issue 

``` shell
cmake .
cmake --build .
```
To clean the build, issue

``` shell
cmake --build . --target clean
```
### TODO make cmake and projucer use the same Build folder
# Testing

TODO add a simple vocal file or grab it.

# License

[Apache-2.0 license](LICENSE)
