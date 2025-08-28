// Saves into a text file the output of a gaussian light beam propagating in a 2D waveguide array

/*
Copyright (C) 2025  Diego Roman-Cortes

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

contact: diego.roman.c@ug.uchile.cl
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

int main(int argc, char* argv[]){    
    // number of points in grid
    int Nx = 700;
    int Ny = 700;
    int Nz = 5000;
    
    // parameters
    double n0 = 1.48; // refraction index of borosilicate 
    double l0 = 730E-9; // wavelenght of light
    double wx = 1.2E-6; // width of the waveguide 
    double wy = 3.0E-6; // height of the waveguide
    double sigma = 8.0E-6; // width of LG-mode
    double l = 0; // azimuthal parameter of LG-mode
    double Lx = 350E-6; // width of the grid
    double Ly = 350E-6; // height of the grid

    double zmax =  50E-3; // propagation distance
    
    // auxiliar variables
    double dx = Lx/(Nx-1);
    double dy = Ly/(Ny-1);
    double dz = zmax/(Nz-1);
    double k0 = 2*M_PI/l0;
    double beta = k0 * n0;
    double xi, yj, r, phi;
   
    //phi = atof(argv[1])*1E-9;
    //printf("%f", phi);
 
    double* dn = malloc(sizeof(double) * Nx * Ny);
    
    // 1D array setup
    double dn1 = 9.5E-4; // contrast of first waveguide

    double d1x = 17E-6; // X separation of waveguides
    double d1y = 18.5E-6; // Y separation of waveguides
    
    // for animation
    int frames = 50;
    int rem, div;
    char filename[10];

    int i, j, k;

    FILE *fp1, *fp2, *fp3;

    //initialization of FFTW
    
    fftw_init_threads(); 
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_complex *aux = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    fftw_plan_with_nthreads(8); 
    
    fftw_plan p_forward = fftw_plan_dft_2d(Nx, Ny, in, out, FFTW_BACKWARD, FFTW_PATIENT);
    fftw_plan p_inverse = fftw_plan_dft_2d(Nx, Ny, aux, in, FFTW_FORWARD, FFTW_PATIENT);

    fp1 = freopen("refractive2d.txt", "w", stdout);
    // shape of refractive index contrast
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            xi = -0.5*Lx + i*dx;
            yj = -0.5*Ly + j*dy;
            
            for(int n=-9; n<10; n++){
                dn[i+Nx*j] += dn1 * tanh(33.0 / (exp(((xi-n*d1x)/wx)*((xi-n*d1x)/wx) + ((yj)/wy)*((yj)/wy))));
            }
            printf("%e\n", dn[i+Nx*j]);
        }
        printf("\n");
    }
    fclose(fp1);
     
    // initial field (gaussian)
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            xi = -0.5*Lx + i*dx;
            yj = -0.5*Ly + j*dy;
            r = sqrt((xi)*(xi) + (yj)*(yj));
            in[i+Nx*j] += (cexp(-r*r/(sigma*sigma))); // hermite-gaussian mode
        }
    }
    // save the input (gaussian) in a text file
    fp2 = freopen("00.txt", "w", stdout);
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            xi = -0.5*Lx + i*dx;
            yj = -0.5*Ly + j*dy;
            printf("%e\n", cabs(in[i+Nx*j])*cabs(in[i+Nx*j]));
        }
        printf("\n");
    }
    fclose(fp2);

    // frequency indices
    int freqidx[Nx + Ny];

    for(i=0; i < Nx/2; i++){
        freqidx[i] = i;
    }
    for(j=0; j < Ny/2; j++){   
        freqidx[Nx+j] = j;
    }
    for(i=Nx/2; i < Nx; i++){
        freqidx[i] =  i-Nx;
    }
    for(j=Ny/2; j < Ny; j++){
        freqidx[Nx+j] =  j-Ny;
    }    
    

    fftw_complex *phase = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nx * Ny);
    for(i = 0; i < Nx; i++){
        for(j = 0; j < Ny; j++){
            phase[i+j*Nx] = cexp(I*dz*( (2*M_PI) * (2*M_PI) * ( (freqidx[i]/Lx) * (freqidx[i]/Lx) + (freqidx[Nx+j]/Ly) * (freqidx[Nx+j]/Ly) )/(4*beta)));
        }
    }

    // main loop

    for(k=1; k <= Nz; k++){

        fftw_execute(p_forward); // 'out' now points towards the DFT of 'in' 
        
        for(i = 0; i < Nx*Ny; i++){
            aux[i] = out[i] * phase[i];
        }

        fftw_execute(p_inverse); // 'in' now points towards the inverse DFT of 'aux'

        for(i = 0; i < Nx * Ny; i++){
            in[i] /= (Nx * Ny); // normalization of FFT
            in[i] *= cexp(-I * k0 * (((n0+dn[i])*(n0+dn[i]))- (n0*n0)) * dz /(2*n0)); // potential operator in real space
        }
        
        fftw_execute(p_forward); // 'out' now points towards the DFT of 'in' 

        for(i = 0; i < Nx*Ny; i++){
            aux[i] = out[i] * phase[i];
        }

        fftw_execute(p_inverse); // 'in' now points towards the inverse DFT of 'aux'
        for(i = 0; i < Nx * Ny; i++){
            in[i] /= (Nx * Ny); // normalization of FFT
        }

        // save to txt
        rem = k % (Nz/frames);
        if(rem == 0){
            div = k / (Nz/frames);
            sprintf(filename, "%02d.txt", div);

            fp3 = freopen(filename, "w", stdout);
        
            for(i = 0; i < Nx; i++){
                for(j = 0; j < Ny; j++){
                    printf("%e\n", cabs(in[i+j*Nx])*cabs(in[i+j*Nx]));
                }
                printf("\n");
            }
            fclose(fp3);
        }

    }
    fftw_cleanup_threads();
    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_inverse);
    fftw_free(in);
    fftw_free(aux); 
    fftw_free(out);
    fftw_free(phase);
    free(dn);
    return 0;
}