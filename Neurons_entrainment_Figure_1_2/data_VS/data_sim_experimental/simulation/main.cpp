// =============================================================
// =============================================================
#include <iostream>
#include <iomanip>  // cout precision
#include <math.h>
#include "VectorMatrix.h"
#include "NervousSystem.h"
using namespace std;
//////////////////////////////////////////////////////////
///////////////// Simulation individual neurons
// Integration Parameters
const double StepSize = 0.00005;
double ttot_noise = 400;
double noise_level = 40.0; // std noise pulses in pA
double pulse_width = 0.0005; // 0.5 ms pulse width

double ttot_sine = 40;
double Iampl = 20.0; // sine amplitude in pA

// Network parameters
const int N = 16;
const int n = 1; // mean numer of synapses going out and in of a neuron
const double pi = 3.141592;
// Files for neurons and network configuration

ifstream rates_noise_file("../data/rates_noise.txt");
ifstream rates_sine_file("../data/rates_sine.txt");
ifstream prc_file("../data/params_prc.txt");
ifstream volt_file("../data/volt.dat");

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int main (int argc, const char* argv[]){
/////////////////      read gmax and Erev from terminal  ////////////////////////////
    double freq = atof(argv[1]);
    //////////////////////////////////////////////////////////
    RandomState rs;
    long randomseed = static_cast<long>(time(NULL));
    cout << "randomseed  " << randomseed << endl;
    rs.SetRandomSeed(randomseed);
    cout.precision(10);
    //////////////////////////////////////////////////////////
    TVector<double> rates_noise(1, N);
    rates_noise_file >> rates_noise;
    rates_noise_file.close();
    //////////////////////////////////////////////////////////
    TVector<double> rates_sine(1, N);
    rates_sine_file >> rates_sine;
    rates_sine_file.close();

    //////////////////////////////////////////////////////////
    NervousSystem gp;           // Connected GPe
    gp.SetCircuitSize(N, 2);
    // Load mean Voltage trace
    gp.LoadVoltage(volt_file);

    //////////////////////////////////////////////////////////
    // load neurons PRCs
    TVector<double> prc_params(1, 7*N);
    prc_file >> prc_params;
    prc_file.close();

    for (int i=1; i<=N; i++){
        for (int p=1; p<=7; p++){
            gp.PrcNeuron(i).PrcParam[p] = prc_params[7*(i-1)+p];
        }
    }
    //// Load PRC interpolation matrix
    gp.LoadPrcVectors();

    // load neurons propierties
    for (int i=1; i<=N; i++){
        gp.SetNeuronCV(i, 0.08);
    }
    //////////////////////////////////////////////
    // Inicialization
    gp.RandomizeCircuitPhase(0.0, 1.0, rs);
    double Iext = 0;
    //////////////////////////////////////////////
    /////  Simulate Noise protocol
    if (freq == 0) {
        for (int i=1; i<=N; i++){
            gp.SetNeuronW(i, rates_noise(i));
        }

        //////////////////////////////////////////////
        ofstream noise("noise.dat");
        //////////////////////////////////////////////

        gp.SetSaveSpikes(false);
        for (double t = 0; t <1; t += StepSize){
            gp.EulerStep(StepSize, t, rs);
        }
        /////////////////// Recording spikes
        gp.SetSaveSpikes(true);
        int Steps_per_pulse = (int)(pulse_width / StepSize);
        int nindx = (int)(Steps_per_pulse/2);
        for (double t = 0; t <ttot_noise; t += StepSize){
            if (nindx == 10){
                Iext = rs.GaussianRandom(0.0, pow(noise_level, 2)); // 40 pA std
                noise << Iext << endl;
                nindx = 0;
            }
            for (int i=1; i<=N; i++)  {
                gp.externalinputs(i) = Iext;
            }
            gp.EulerStep(StepSize, t, rs);
            nindx +=1;
        }
        ///////
        noise.close();
    //////////////////////////////////////////////
    /////  Simulate Sine protocol
    } else {
        for (int i=1; i<=N; i++){
            gp.SetNeuronW(i, rates_sine(i));
        }
        //////////////////////////////////////////////
        gp.SetSaveSpikes(false);
        for (double t = 0; t <1.0; t += StepSize){
            Iext = Iampl*sin(2*pi*t*freq);
            for (int i=1; i<=N; i++)  {
                gp.externalinputs(i) = Iext;
            }
            gp.EulerStep(StepSize, t, rs);
        }
        /////////////////// Recording spikes
        gp.SetSaveSpikes(true);
        for (double t = 0; t <ttot_sine; t += StepSize){
            Iext = Iampl*sin(2*pi*t*freq);
            for (int i=1; i<=N; i++)  {
                gp.externalinputs(i) = Iext;
            }
            gp.EulerStep(StepSize, t, rs);
        }
    }

    ///////
}
