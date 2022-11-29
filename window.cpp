#include <SDL.h>
#include <SDL_image.h>
#include "window.h"

ImageData::ImageData(SDL_Surface* image_surface, int x_pos, int y_pos,
	int height, int width){

}

Window::Window(int height, int width, std::string title): 
	main_window(nullptr), main_surface(nullptr), is_initialized(false) {

	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		err_string = SDL_GetError();
		return;
	}

	char* title_d = const_cast<char*>(title.data());
	main_window = SDL_CreateWindow(title_d, SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED, height, width, SDL_WINDOW_SHOWN);

	if(main_window == nullptr){
		err_string = SDL_GetError();
		return;
	}

	main_surface = SDL_GetWindowSurface(main_window);

	SDL_PixelFormat form = {.format = SDL_PIXELFORMAT_ARGB8888 };
	if(SDL_PixelFormatEnumToMasks(SDL_PIXELFORMAT_ARGB8888,
		&(form.BitsPerPixel), &(form.Rmask), &(form.Gmask), &(form.Mmask),
		&(form.Amask)) != SDL_TRUE){
		err_string = SDL_GetError();
	}

	is_initialized = true;
}

Window& add_image(std::string filename){
	SDL_Surface* image_surface  = IMG_Load(const_cast<char*>(filename.c_str()));
	if(image_surface->format != SDL_PIXELFORMAT_ARGB8888)
		image_surface = SDL_ConvertSurfaceFormat(image_surface,
			SDL_PIXELFORMAT_ARGB8888, 0);
	ImageData* d = new ImageData
}

template<std::size_t R, std::size_t C>
Window& Window::add_image(std::array<unsigned char, R*C> bytes, int x, int y){
	ImageData* d = new ImageData(bytes.data(),x,y,R,C);
	images.push_back(d);
}

Window::~Window(){
	for(auto x : images){
		SDL_FreeSurface(x->image_surface);
		free x;
	}
}