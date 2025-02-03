//////////////////////////////////////////////////////////////////////
//
//	University of Leeds
//	COMP 5892M Advanced Rendering
//	User Interface for Coursework
//
//	September, 2020
//
//  -----------------------------
//  Raytrace Render Widget
//  -----------------------------
//
//	Provides a widget that displays a fixed image
//	Assumes that the image will be edited (somehow) when Render() is called
//  
////////////////////////////////////////////////////////////////////////

// include guard
#ifndef RAYTRACE_RENDER_WIDGET_H
#define RAYTRACE_RENDER_WIDGET_H

#include <vector>
#include <mutex>
#include <atomic>

// and include all of our own headers that we need
#include "ThreeDModel.h"
#include "RenderParameters.h"
#include "Scene.h"

class Raytracer 										
	{ 
	
	private:	
	// the geometric object to be rendered
    std::vector<ThreeDModel> *texturedObjects;
	// the render parameters to use
	RenderParameters *renderParameters;
	// An image to use as a framebuffer
    //A friendly Scene representation that we control
    Scene raytraceScene;

	public:
	// constructor
	Raytracer
			(
	 		// the geometric object to show
            std::vector<ThreeDModel> 		*newTexturedObject,
			// the render parameters to use
			RenderParameters 	*newRenderParameters
			);
	
	// destructor
	~Raytracer();
	
	void resize(int w, int h);
	void stopRaytracer();
	RGBAImage frameBuffer;

	protected:

	// called every time the widget needs painting

	public:

    // routine that generates the image
    void Raytrace();
    //threading stuff
    void RaytraceThread();

	Ray calculateRay(int pixelx, int pixely, bool perspective);

	Homogeneous4 TraceAndShadeWithRay(Ray r, int bounces, float reflectionFactor, float currentIOR = 1.0f);

	Homogeneous4 reflectionShading(Ray ray, Cartesian3 normal, Cartesian3 point, Homogeneous4 color, float reflectivity, float reflectionFactor, int bounces);

	Homogeneous4 interpolatedShading(Scene::CollisionInfo ci, Ray r);

	Homogeneous4 shadowShading(Scene::CollisionInfo ci, Light* l, Cartesian3 currentPoint, Cartesian3 normal, Cartesian3 bc, Homogeneous4 phongColor);

	Ray reflectRay(Ray r, Cartesian3 normal, Cartesian3 intersectionPoint);

	// routine to refract a ray
	// returns true if the ray is refracted, false if there is total internal reflection
	// hence the direction is taken as a reference
	// normal is taken as a reference too because it is used to offset the origin of this new ray 
	// and the normal changes the calculation if we are refracting from air to a material or vice versa
	bool refractRay(Ray &incidentRay, Cartesian3 &intersectionPoint, Cartesian3 &normal, float &currentIOR, float &ior, Cartesian3 &direction);

	float schlickApproximation( float cosTheta, float ior1, float ior2)
	{
		float R0 = pow((ior1 - ior2) / (ior1 + ior2), 2.0f);
		return R0 + (1.0f - R0) * pow(1.0f - cosTheta, 5.0f);
	}


    private:

	std::atomic<bool> raytracingRunning;
	std::atomic<bool> restartRaytrace;

	}; // class RaytraceRenderWidget

#endif
