#define SDL_MAIN_HANDLED

//standard libraries
#include <iostream>
#include <cstdint>
#include <array>

//Media libraries
#include <SDL2/SDL.h>

//new libraries
#include <c++/window.hpp>

const size_t SCREEN_WIDTH = 1000;
const size_t SCREEN_HEIGHT = 512;

struct pixels {
	//Little endian pain
	unsigned char b;
	unsigned char g;
	unsigned char r;
	unsigned char a;
};


#if defined BW
//make sure that the pixel format is ARGB8888 little-endian
__global__ void testKernel(pixels* d_inpixels, pixels* d_outpixels, int height, int width){
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

#elif defined BLUR
//static variable blur_size
__global__ void testKernel(pixels* d_inpixels, pixels* d_outpixels, int height, int width){
	static int blur_size = 3;
	size_t t_row = blockIdx.x * blockDim.x + threadIdx.x;
	size_t t_col = blockIdx.y * blockDim.y + threadIdx.y;
	int a,r,g,b;
	a = r = g = b = 0;
	int count_values = 0;
	for(int row_off = - blur_size; row_off <= blur_size ; row_off++)
		for(int col_off = -blur_size; col_off <= blur_size ; col_off++){ //man I am lazy to just create new variables
			//sampling coordinates
			int s_row = t_row + row_off;
			int s_col = t_col + col_off;

			//check if under bounds
			if(s_row < 0 || s_col < 0 || s_row >= height || s_col >= width)
				continue;

			//how many we are sampling?
			count_values++;

			pixels* curr = d_inpixels + s_row*width + s_col;
			a += curr->a;
			r += curr->r;
			g += curr->g;
			b += curr->b;
		}

	//average out and store the output pixel values
	pixels* out = d_outpixels + t_row*width + t_col;
	out->a = a/count_values;
	out->r = r/count_values;
	out->g = g/count_values;
	out->b = b/count_values;
}

#else 
#error "Define BLUR or BW"
#endif

int main(){
	//hardcoding these values for demo
	//TODO: make the resolution to-fit algorithmically
	constexpr size_t row = 512, col = 512;
	constexpr size_t image_size = 4*row*col;

	Window win(SCREEN_HEIGHT,SCREEN_WIDTH);
	win.add_image("lena_color.tiff", 0,0,512,500);

	win.show();
	SDL_Delay(1000);

	//image array as they're added, will be in SDL_PIXELFORMATARGB8888 format
	auto images = win.get_images();

	//2^4*2^5 = 2^9 = 512
	dim3 dimGrid(16,16,1);
	dim3 dimBlock(32,32,1);

	//device memory
	pixels* d_in, *d_out;

	//device memory allocations and initialization
	cudaMalloc((void**) &d_in, image_size);
	cudaMemcpy((void*) d_in, images[0]->image_surface->pixels,image_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_out, image_size);

	//start the kernel
	//colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(d_in, d_out, row, col);
	testKernel<<<dimGrid, dimBlock>>>(d_in, d_out, row, col);
	cudaDeviceSynchronize();

	//using raw continuous bytes for allocation, C style code but eh
	auto bytes = new unsigned char[image_size];
	cudaMemcpy(bytes,d_out,image_size, cudaMemcpyDeviceToHost);
	win.add_image(bytes,512,512,500,0,512,500);
	win.show();

	cudaFree(d_in); cudaFree(d_out);

	SDL_Delay(10000);

	delete[] bytes;
	return 0;
}