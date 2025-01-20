#include "Triangle.h"
#include <math.h>
#include <algorithm>

Triangle::Triangle()
{
    triangle_id = -1;
    shared_material= nullptr;
}


void Triangle::validate(int id){
    triangle_id = id;
}

bool Triangle::isValid(){
    return triangle_id != -1;
}

float Triangle::intersect(Ray r)
{
    float epsilon = 0.001f;

    Cartesian3 v0 = verts[0].Point(); //p
    Cartesian3 v1 = verts[1].Point();
    Cartesian3 v2 = verts[2].Point();


    // compute plane's normal
    Cartesian3 v0v1 = v1 - v0; //u
    v0v1.unit();
    Cartesian3 v0v2 = v2 - v0;
    v0v2.unit();
    Cartesian3 N = v0v1.cross(v0v2); // N
    N.unit();
    Cartesian3 w = N.cross(v0v1);    // w
    w.unit();

    float t = (v0 - r.origin).dot(N) / r.direction.dot(N);

    if (r.direction.dot(N) == 0.0)
    {
        return -1;
    }

    if (t > 0.0f)
    {
        // printf("t %f\n",t);
        Cartesian3 o = r.origin + r.direction * (t);

        // printf("o %f\n",o.length());

        // convert to PCS
        Cartesian3 planarO = o - v0;
        Cartesian3 planarV0 = v0 - v0;
        Cartesian3 planarV1 = v1 - v0;
        Cartesian3 planarV2 = v2 - v0;

        planarO = Cartesian3(planarO.dot(v0v1), planarO.dot(w), planarO.dot(N));
        planarV0 = Cartesian3(planarV0.dot(v0v1), planarV0.dot(w), planarV0.dot(N));
        planarV1 = Cartesian3(planarV1.dot(v0v1), planarV1.dot(w), planarV1.dot(N));
        planarV2 = Cartesian3(planarV2.dot(v0v1), planarV2.dot(w), planarV2.dot(N));

        // return (o - r.origin).length();

        // half plane test
        Cartesian3 PCSV0V1 = planarV1 - planarV0;
        Cartesian3 PCSV1V2 = planarV2 - planarV1;
        Cartesian3 PCSV2V0 = planarV0 - planarV2;

        Cartesian3 PCSOV0 = planarO - planarV0;
        Cartesian3 PCSOV1 = planarO - planarV1;
        Cartesian3 PCSOV2 = planarO - planarV2;


        Cartesian3 half_PCSV0V1 = Cartesian3(-PCSV0V1.y, PCSV0V1.x, 0).unit();
        Cartesian3 half_PCSV1V2 = Cartesian3(-PCSV1V2.y, PCSV1V2.x, 0).unit();
        Cartesian3 half_PCSV2V0 = Cartesian3(-PCSV2V0.y, PCSV2V0.x, 0).unit();

        float dota = PCSOV0.dot(half_PCSV0V1);
        float dotb = PCSOV1.dot(half_PCSV1V2);
        float dotc = PCSOV2.dot(half_PCSV2V0);

        if (PCSOV0.dot(half_PCSV0V1) < 0 || PCSOV1.dot(half_PCSV1V2) < 0 || PCSOV2.dot(half_PCSV2V0) < 0)
        {
            // printf("t %f\n",t);
            return -1.0f;
        }
    }

    return t;

}

Cartesian3 Triangle::baricentric(Cartesian3 o)
{
    Cartesian3 bc;

    // compute normal at o using barycentric coordinates
    Cartesian3 A = verts[0].Point();
    Cartesian3 B = verts[1].Point();
    Cartesian3 C = verts[2].Point();

    Cartesian3 BC = C - B;
    Cartesian3 CA = A - C;
    Cartesian3 AB = B - A;

    Cartesian3 PA = o-A;
    Cartesian3 PB = o-B;
    Cartesian3 PC = o-C;

    float areaABC = (AB.cross(CA).length());
    float areaPBC = (BC.cross(PB).length());
    float areaPCA = (CA.cross(PC).length());
    float areaPAB = (AB.cross(PA).length());
    
    float alpha = areaPBC / areaABC;
    float beta = areaPCA / areaABC;
    float gamma = areaPAB / areaABC;

    bc = Cartesian3(alpha, beta, gamma);

    return bc;

}

Homogeneous4 Triangle::phongShading(Homogeneous4 lightPos, Homogeneous4 lightColor, Cartesian3 bc)
{
     
    Homogeneous4 color;

    // interpolate normals
    Cartesian3 Na = normals[0].Vector();
    Cartesian3 Nb = normals[1].Vector();
    Cartesian3 Nc = normals[2].Vector();

    Cartesian3 normal = (Na * bc.x + Nb * bc.y + Nc * bc.z).unit();

    Cartesian3 position = verts[0].Point() * bc.x + verts[1].Point() * bc.y + verts[2].Point() * bc.z;
    Cartesian3 l = (lightPos.Point() - position).unit();
    Cartesian3 LIGHT_COLOR = lightColor.Vector();

    // ambient
    Cartesian3 ambient = Cartesian3(shared_material->ambient.x * LIGHT_COLOR.x , shared_material->ambient.y * LIGHT_COLOR.y, shared_material->ambient.z * LIGHT_COLOR.z);

    // diffuse
    float cosTheta = std::clamp(normal.dot(l), 0.0f, 1.0f);
    Cartesian3 diffuse = (shared_material->diffuse);
    diffuse = Cartesian3(diffuse.x * LIGHT_COLOR.x , diffuse.y * LIGHT_COLOR.y , diffuse.z * LIGHT_COLOR.z);
    diffuse = diffuse * cosTheta;

    // specular
    Cartesian3 e = (Cartesian3(0,0,0) - position).unit();
    Cartesian3 B = (l + e).unit();
    float cosB = std::clamp(normal.dot(B), 0.0f, 1.0f);

    cosB = std::clamp(pow(cosB, shared_material->shininess), 0.0f, 1.0f);

    cosB = cosB * cosTheta * (shared_material->shininess + 2) / (2*M_PI);

    Cartesian3 specular = (shared_material->specular);
    specular = Cartesian3(specular.x * LIGHT_COLOR.x , specular.y * LIGHT_COLOR.y , specular.z * LIGHT_COLOR.z);
    specular = specular * cosB;

    color = diffuse + specular + ambient;

    return color;

}


Homogeneous4 Triangle::shadowShading(Homogeneous4 lightColor)
{
    Homogeneous4 color;

    Cartesian3 LIGHT_COLOR = lightColor.Vector();

    // ambient
    Cartesian3 ambient = Cartesian3(shared_material->ambient.x * LIGHT_COLOR.x , shared_material->ambient.y * LIGHT_COLOR.y, shared_material->ambient.z * LIGHT_COLOR.z);

    color = ambient;

    return color;
}