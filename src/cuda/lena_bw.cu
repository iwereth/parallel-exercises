#define SDL_MAIN_HANDLED

//standard libraries
#include <iostream>
#include <cstdint>
#include <array>

//Media libraries
#include <SDL2/SDL.h>

//new libraries
#include <c++/window.hpp>

const size_t SCREEN_WIDTH = 1024;
const size_t SCREEN_HEIGHT = 512;

struct pixels {
	unsigned char b;
	unsigned char g;
	unsigned char r;
	unsigned char a;
};

//making sure that the pixel format is ARGB8888
__global__ void colorToGreyscaleConversion(pixels* d_inpixels, pixels* d_outpixels, int height, int width){
	size_t t_row = blockIdx.x * blockDim.x + threadIdx.x;
	size_t t_col = blockIdx.y * blockDim.y + threadIdx.y;

	if(t_row < height && t_col < width){
		size_t offset = t_row * width + t_col;

		pixels in = d_inpixels[offset];
		unsigned char gray_value = 
			in.r * 0.21 + in.g * 0.72 + in.b * 0.07;

		d_outpixels[offset].a = 0xff;
		d_outpixels[offset].r = d_outpixels[offset].g = d_outpixels[offset].b = gray_value;
	}
}

int main(){
	Window win(SCREEN_HEIGHT,SCREEN_WIDTH);
	constexpr size_t row = 512, col = 512;
	constexpr size_t image_size = sizeof(pixels)*row*col;
	win.add_image("lena_color.tiff", 0,0,row,col);
	win.show();
	SDL_Delay(1000);
	auto images = win.get_images();
	
	dim3 dimGrid(16,16,1);
	dim3 dimBlock(32,32,1);

	pixels* d_in, *d_out;

	cudaMalloc((void**) &d_in, image_size);
	cudaMemcpy((void*) d_in, images[0]->image_surface->pixels,image_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_out, image_size);

	colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(d_in, d_out, row, col);
	cudaDeviceSynchronize();

	auto bytes = new unsigned char[image_size];

	cudaMemcpy(bytes,d_out,image_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in); cudaFree(d_out);

	win.add_image(bytes, 512,0,512,512);

	win.show();

	SDL_Delay(10000);
	return 0;
}