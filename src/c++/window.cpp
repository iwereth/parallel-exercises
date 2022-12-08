//standard libraries
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>

//new libraries
#include <c++/window.hpp>

ImageData::ImageData(SDL_Surface* image_surface, int x_pos, int y_pos,
	int height, int width, bool source_sdl) : image_surface(image_surface), x_pos(x_pos),
	y_pos(y_pos), height(height), width(width), source_sdl(source_sdl){}


void Window::error(const std::string& str){
	std::cout<<str<<std::endl;
	exit(1);
}

Window::Window(int height, int width, std::string title): 
	main_window(nullptr), main_surface(nullptr), is_initialized(false) {

	if(SDL_Init(SDL_INIT_VIDEO) < 0){
		err_string = SDL_GetError();
		return;
	}

	char* title_d = const_cast<char*>(title.data());
	main_window = SDL_CreateWindow(title_d, SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);

	if(main_window == nullptr){
		err_string = SDL_GetError();
		return;
	}

	main_surface = SDL_GetWindowSurface(main_window);

	is_initialized = true;
}

Window& Window::add_image(const std::string& filename, int x, int y, int h, int w){

	SDL_Surface* image_surface  = IMG_Load(const_cast<char*>(filename.c_str()));

	if(image_surface->format->format != SDL_PIXELFORMAT_ARGB8888){
		image_surface = SDL_ConvertSurfaceFormat(image_surface,SDL_PIXELFORMAT_ARGB8888, 0);
		if(image_surface == NULL){
			error("Error in add_image : Cannot convert surface format");
		}
	}

	ImageData* d = new ImageData(image_surface, x, y, h, w, true);
	images.push_back(d);
	return *this;
}


//a fked up idea I had
/*template<std::size_t R, std::size_t C>
Window& Window::add_image(const std::array<unsigned char, R*C>& bytes, int x, int y){
	unsigned char* pixels = new unsigned char[R*C];
	pixels = std::memcpy(pixels, bytes.data(), 4*R*C*sizeof(unsigned char));

	SDL_Surface* image_surface = SDL_CreateRGBSurfaceWithFormatFrom(
		static_cast<void*>(pixels), C, R, 32, 4*C, SDL_PIXELFORMAT_ARGB8888);
	if(image_surface == NULL){
		std::cerr<<"Error in add_image: Cannot convert surface from pixels\n";
		exit(1);
	}	

	images.push_back(new ImageData(image_surface,x,y,R,C));
	return *this;
}*/

Window& Window::add_image(const unsigned char* bytes , int x , int y, int h, int w){
	unsigned char* pixels = new unsigned char[4*h*w];
	std::memcpy(pixels, bytes, 4*h*w*sizeof(unsigned char));

	auto format = SDL_PIXELFORMAT_ARGB8888;

	SDL_Surface* image_surface = SDL_CreateRGBSurfaceWithFormatFrom(
		static_cast<void*>(pixels), w, h, SDL_BITSPERPIXEL(format), 
		w*SDL_BYTESPERPIXEL(format), SDL_PIXELFORMAT_ARGB8888);

	if(image_surface == NULL){
		error("Error in add_image: Cannot convert to surface from pixels");
	}

	images.push_back(new ImageData(image_surface,x,y,h,w,false));
	return *this;
}

void Window::show(void){
	for(auto image : images){
		SDL_Rect rect = {image->x_pos, image->y_pos, image->width, image->height};
		if(SDL_BlitScaled(image->image_surface, NULL, main_surface, &rect) < 0){
			error(SDL_GetError());
		}
	}
	SDL_UpdateWindowSurface(main_window);
}

std::vector<ImageData*>& Window::get_images(void){
	return images;
}

Window::~Window(){
	for(auto x : images){
		unsigned char* t = nullptr;
		if(x->source_sdl == false)
			auto t = static_cast<unsigned char*> (x->image_surface->pixels);
		SDL_FreeSurface(x->image_surface);
		delete x;
		delete[] t;
	}
	SDL_FreeSurface(main_surface);
	SDL_DestroyWindow(main_window);
	SDL_Quit();
}
