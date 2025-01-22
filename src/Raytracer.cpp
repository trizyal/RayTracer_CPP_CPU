//////////////////////////////////////////////////////////////////////
//
//  University of Leeds
//  COMP 5892M Advanced Rendering
//  User Interface for Coursework
////////////////////////////////////////////////////////////////////////


#include <math.h>
#include <thread>
#include <random>
#include <omp.h>
#include <algorithm>
// include the header file
#include "Raytracer.h"

#define N_THREADS 16
#define N_LOOPS 600
#define N_BOUNCES 10
#define TERMINATION_FACTOR 0.35f

// constructor
Raytracer::Raytracer(std::vector<ThreeDModel> *newTexturedObject, RenderParameters *newRenderParameters):
    texturedObjects(newTexturedObject),
    renderParameters(newRenderParameters),
    raytraceScene(texturedObjects,renderParameters)
    { 
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        restartRaytrace = false;
        raytracingRunning = false;
    }     


Raytracer::~Raytracer()
    { 
    // all of our pointers are to data owned by another class
    // so we have no responsibility for destruction
    }                                                                  


// called every time the widget is resized
void Raytracer::resize(int w, int h)
    { // RaytraceRenderWidget::resizeGL()
    // resize the render image
    
    frameBuffer.Resize(w, h);
    // frameBuffer.clear(RGBAValue(125.0f, 125.0f, 125.0f, 255.0f));
    } // RaytraceRenderWidget::resizeGL()
    
void Raytracer::stopRaytracer() {
    restartRaytrace = true;
    while (raytracingRunning) {
        std::chrono::milliseconds timespan(10);
        std::this_thread::sleep_for(timespan);
    }
    restartRaytrace = false;
}

inline
float linear_from_srgb(std::uint8_t aValue) noexcept
{
    float const fvalue = float(aValue) / 255.f;

    if (fvalue < 0.04045f)
        return (1.f / 12.92f) * fvalue;

    return std::pow((1.f / 1.055f) * (fvalue + 0.055f), 2.4f);

}

inline
std::uint8_t linear_to_srgb(float aValue) noexcept
{
    if (aValue < 0.0031308f)
        return std::uint8_t(255.f * 12.92f * aValue + 0.5f);
    return std::uint8_t(255.f * (1.055f * std::pow(aValue, 1.f / 2.4f) - 0.055f) + 0.5f);
}




void Raytracer::RaytraceThread()
{    
    for (int j = 0 ; j < frameBuffer.height ; j++)
    {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0 ; i < frameBuffer.width ; i++)
        {
            Ray cameraRay = calculateRay(i, j, !renderParameters->orthoProjection);

            Homogeneous4 color = Homogeneous4(0.0f, 0.0f, 0.0f, 1.0f);

            // interpoltation coloring
            Scene::CollisionInfo ci = raytraceScene.closestTriangle(cameraRay);
            if (ci.t > -0.01f)
            {
                if (renderParameters->interpolationRendering)
                {
                    // just normals shading
                    color =  interpolatedShading(ci, cameraRay);
                }
                else if (renderParameters->phongEnabled)
                {
                    // raytracing proper
                    color = TraceAndShadeWithRay(cameraRay, N_BOUNCES, 1.0f);
                }
                else
                {  
                    // no shading, just white if ray hit something
                    color = Homogeneous4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
            else 
            {
                // no intersection
                color = Homogeneous4(0.0f, 0.0f, 0.0f, 1.0f);
            }
            
            // set the color
            frameBuffer[j][i] = RGBAValue{linear_to_srgb(color.x), linear_to_srgb(color.y), linear_to_srgb(color.z), 255};
        }

        if (restartRaytrace)
        {
            raytracingRunning = false;
            return;
        }
    }
    raytracingRunning = false;
}




Homogeneous4 Raytracer::TraceAndShadeWithRay(Ray r, int bounces, float reflectionFactor, float currentIOR)
{
    Scene::CollisionInfo ci = raytraceScene.closestTriangle(r);

    if (ci.t > -0.01f)
    {
        // this the shared bit
        Cartesian3 o = r.origin + r.direction * ci.t;
        Cartesian3 bc = ci.tri.baricentric(o);

        Homogeneous4 phongColor = ci.tri.shared_material->emissive;

        Cartesian3 Na = ci.tri.normals[0].Vector();
        Cartesian3 Nb = ci.tri.normals[1].Vector();
        Cartesian3 Nc = ci.tri.normals[2].Vector();
        Cartesian3 normal = (Na * bc.x + Nb * bc.y + Nc * bc.z).unit();

        Cartesian3 currentPoint = ci.tri.verts[0].Point() * bc.x + ci.tri.verts[1].Point() * bc.y + ci.tri.verts[2].Point() * bc.z;

        for (Light* l : renderParameters->lights)
        {
            // Cartesian3 currentPoint = ci.tri.verts[0].Point() * bc.x + ci.tri.verts[1].Point() * bc.y + ci.tri.verts[2].Point() * bc.z;

            if (renderParameters->shadowsEnabled)
            {
                // shadows are enabled and we need to check if we are in shadow
                phongColor = shadowShading(ci, l, currentPoint, normal, bc, phongColor);
            }
            else
            {
                // shadows are disabled
                Matrix4 modelview = raytraceScene.getModelview();
                Homogeneous4 lightPos = modelview * l->GetPositionCenter();
                Homogeneous4 lightColor = l->GetColor();
                phongColor = phongColor + ci.tri.phongShading(lightPos, lightColor, bc);
            }

            // if (renderParameters->reflectionEnabled)
            // {
            //     // reflection is enabled, this is the recursive part
            //     phongColor = reflectionShading(r, normal, currentPoint, phongColor, ci.tri.shared_material->reflectivity, reflectionFactor, bounces);
            // }

            if (renderParameters->refractionEnabled)
            {
                // refraction is enabled, this is the recursive part
                phongColor = reflectionShading(r, normal, currentPoint, phongColor, ci.tri.shared_material->reflectivity, reflectionFactor, bounces);

                Ray refractedRay = refractRay(r, normal, currentPoint, ci.tri.shared_material->indexOfRefraction);
                Homogeneous4 refractedColor = TraceAndShadeWithRay(refractedRay, bounces - 1, reflectionFactor * ci.tri.shared_material->reflectivity, ci.tri.shared_material->indexOfRefraction);
                phongColor = phongColor + refractedColor;
            }
        }
        // just sanity check the color
        phongColor.x = std::clamp(phongColor.x, 0.0f, 1.0f);
        phongColor.y = std::clamp(phongColor.y, 0.0f, 1.0f);
        phongColor.z = std::clamp(phongColor.z, 0.0f, 1.0f);
        return phongColor;
    }
    else 
    {
        // no intersection
        return Homogeneous4(0.0f, 0.0f, 0.0f, 1.0f);
    }

}




Homogeneous4 Raytracer::reflectionShading(Ray ray, Cartesian3 normal, Cartesian3 point, Homogeneous4 color, float reflectivity, float reflectionFactor, int bounces)
{
    if (reflectivity <= 0.0f)
        return color;

    if (bounces > 0)
    {
        Ray reflectedRay = reflectRay(ray, normal, point);
        Homogeneous4 reflectedColor = TraceAndShadeWithRay(reflectedRay, bounces - 1, reflectionFactor * reflectivity);
        color = (color*(1-reflectivity) + reflectedColor) * reflectionFactor;
    }
    else if (bounces == 0)
        color = Homogeneous4(0.0f, 0.0f, 0.0f, 0.0f);

    return color;
}




Homogeneous4 Raytracer::shadowShading(Scene::CollisionInfo ci, Light* l, Cartesian3 currentPoint, Cartesian3 normal, Cartesian3 bc, Homogeneous4 phongColor)
{
    // for this light, are we in shadow?
    float epsilon = 0.01f;
    Cartesian3 lp = raytraceScene.getModelview() * l->GetPositionCenter().Point();
    Ray shadowRay =  Ray((currentPoint+epsilon*normal), ( lp - currentPoint).unit(), Ray::secondary);

    Scene::CollisionInfo ci_shadow = raytraceScene.closestTriangle(shadowRay);

    if (ci_shadow.t > 0.0f && ci_shadow.tri.isValid() && !ci_shadow.tri.shared_material->isLight())
    {
        // we are in shadow
        Homogeneous4 ambient = ci.tri.shadowShading(l->GetColor());
        phongColor = phongColor + ambient;
    }
    else
    {
        // we are not in shadow
        Matrix4 modelview = raytraceScene.getModelview();
        Homogeneous4 lightPos = modelview * l->GetPositionCenter();
        Homogeneous4 lightColor = l->GetColor();
        phongColor = phongColor + ci.tri.phongShading(lightPos, lightColor, bc);
    }
    
    return phongColor;
}




Ray Raytracer::reflectRay(Ray r, Cartesian3 normal, Cartesian3 intersectionPoint)
{
    Cartesian3 reflection = r.direction - (normal * 2.0f * r.direction.dot(normal));
    return Ray(intersectionPoint+normal*0.001f, reflection, Ray::secondary);
}

Ray Raytracer::refractRay(Ray r, Cartesian3 normal, Cartesian3 intersectionPoint, float ior)
{
    auto I = r.direction;
    auto costheta = normal.dot(I);

    auto T = I * 1/ior + (1/ior * costheta - sqrt(1 - 1/ior * 1/ior * (1 - costheta * costheta))) * normal;

    return Ray(intersectionPoint - normal * 0.001f, T, Ray::secondary);
}



    // routine that generates the image
void Raytracer::Raytrace()
{ // RaytraceRenderWidget::Raytrace()
    stopRaytracer();
    //To make our lifes easier, lets calculate things on VCS.
    //So we need to process our scene to get a triangle soup in VCS.
    raytraceScene.updateScene();
    frameBuffer.clear(RGBAValue(0.0f, 0.0f, 0.0f,1.0f));
    std::thread raytracingThread(&Raytracer::RaytraceThread,this);
    raytracingThread.detach();
    raytracingRunning = true;
} // RaytraceRenderWidget::Raytrace()
    

Ray Raytracer::calculateRay(int pixelx, int pixely, bool perspective)
{
    // std::cout<<"pixely "<<pixely<<std::endl;
    // std::cout<<"pixelx "<<pixelx<<std::endl;

    float width = frameBuffer.width;
    float height = frameBuffer.height;

    float x_ndcs = (pixelx/static_cast<float>(width) - 0.5) * 2;
    float y_ndcs = (pixely/static_cast<float>(height) - 0.5) * 2;

    float aspect_ratio = width/static_cast<float>(height);

    float fov_factor = tanf(renderParameters->fov / 2);

    float x , y;

    if (perspective)
    {
        x = x_ndcs * aspect_ratio * fov_factor;
        y = y_ndcs * fov_factor;
        Cartesian3 origin = Cartesian3(0,0,0);
        Cartesian3 direction = Cartesian3(x,y,1);

        return Ray(origin,direction,Ray::primary);        
    }
    else
    {
        if (aspect_ratio > 1)
        {
            x = x_ndcs * aspect_ratio;
            y = y_ndcs;
        }
        else 
        {
            x = x_ndcs;
            y = y_ndcs/aspect_ratio;
        }
        Cartesian3 origin = Cartesian3(x,y,0);
        Cartesian3 direction = Cartesian3(0,0,1);
        // direction = origin-direction;
        direction = direction.unit();
        return Ray(origin,direction,Ray::primary);
    }

}



Homogeneous4 Raytracer::interpolatedShading(Scene::CollisionInfo ci, Ray r)
{
    Cartesian3 o = r.origin + r.direction * ci.t;
    Cartesian3 bc = ci.tri.baricentric(o);
    Cartesian3 Na = ci.tri.normals[0].Vector();
    Cartesian3 Nb = ci.tri.normals[1].Vector();
    Cartesian3 Nc = ci.tri.normals[2].Vector();
    Cartesian3 normal = Na * bc.x + Nb * bc.y + Nc * bc.z;
    normal.unit();

    Homogeneous4 color = Homogeneous4(abs(normal.x), abs(normal.y), abs(normal.z), 1.0f);
    return color;
}
