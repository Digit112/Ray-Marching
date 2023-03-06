# Ray-Marching
Ray Marching rendering engine. I use it to render fractals.

Ray marching is a rendering technique whereby a ray is spawned and we call a distance function to get the distance from the ray to all of the objects in the scene. We then advance the ray that far, knowing that it won't pass through the surface of any object. We continue to step the ray forward until the distance estimate is very small, at which point we say that the ray has collided with an object. Of course, for this to work, all objects must have well-defined distance estimation functions. This is easy for a number of primitives such as spheres and rectangular prisms, as well as some more exotic shapes.

## Features of Ray Marching

- Very close to objects, the distance estimate is very small. That's pretty obvious, but this allows us to keep track of when a ray nearly hits an object and we can use that information to create soft shadows, as well as add a glowing effect to objects at almost no extra cost.

- My library does not currently do this, but by replacing the min function used to combine distance estimations with something else, you can do all sorts of things such as generate the intersection of two objects or blend the objects together at very little cost.

- It is also trivial to define a distance estimation function that is the result of apply transforms repeatedly to an object. As a result, self-similar fractals are best rendered through ray marching. A great example of this would be the Sierpinski tetrahedron, a 3D generalization of the Sierpinski triangle or Sierpinski gasket.

- In addition to self-similar fractals, many fractals have well-defined distance estimators. for these fractals, ray marching is also the best choice. The best-known example fitting this description is the Mandelbulb, but there is an infinite family of distance estimated fractals that can be smoothly blended between while rendering in real time.

## What My Library Can Do

By far my favorite feature of this library is that it is compatible both with NVIDIA GPUs and the CPU. The same source code file (ray_march.hpp) is included either way. If you intend to render for the GPU, put the following preprocessor directive before the include line:

    #define GPU_ENABLED

and compile with `nvcc` instead of your normal compiler. You will need to install the CUDA toolkit and Visual Studio first.

All distance estimators and shaders are 100% reusable between the CPU and GPU. There is no need to define two versions of a function like this unless, for example, you intend to re-write one version to be more efficient on that device.

This library can generate the normal vector at any point in space by taking the gradient of six distance estimates. With that value, we can calculate reflections, refractions, and do just about any other shading trickery. At current, my library contains an implementation of the Phong reflection model, allowing objects and light sources to have distinct diffuse and specular shading. Objects can be very flat or shiny, lights can cast very soft or very hard shadows, and so on. Lights and objects can both have their associated colors, and lights can be point lights or infinitely distant.

The library supports arbitrary numbers of objects and lights per scene. Objects can have "parameters", which in essence is just an array of floats that are user defined, can be changed in between renders at no cost (because there is no baking), and can be accessed and modified by both the distance-estimation function and the shading function at any time.

In addition to having distinct distance estimators, objects can also have distinct shading functions. Both are implemented with function pointers. This will, in theory, allow glass and mirror shaders to be added to the rendering engine, although I don't intend to extend the library very much in the future. There are quite a few things I would love to change, however.