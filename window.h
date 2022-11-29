#include <SDL.h>
#include <SDL_image.h>
#include <string>
#include <vector>

class ImageData{
public:
	unsigned char* pix_data;
	int x_pos, y_pos, height, width;
	bool is_ok;

	ImageData(SDL_Surface* image_surface, int x_pos, int y_pos,
		int height, int width);
};

class Window{
	SDL_Window* main_window;
	SDL_Surface* main_surface;
	bool is_initialized;
	const char* err_string;
	std::vector<ImageData*> images;
	SDL_PixelFormat form;

public:
	Window(int height, int width, std::string title = "CUDA Test");

	template <std::size_t R, std::size_t C>
	Window& add_image(std::array<unsigned char, R*C> bytes, int x, int y, int h, int w);

	~Window();
};