
#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif


#include "hnswlib.h"
namespace hnswlib {
	using namespace std;
	static float
		InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr)
	{
        size_t qty = *((size_t *)qty_ptr);
		float res = 0;
		for (int i = 0; i < qty; i++) {
			float t = -((float*)pVect1)[i] * ((float*)pVect2)[i];
			res += t;
		}
		return (res);

	}
	
	class L2Space : public SpaceInterface<float> {
		
		DISTFUNC<float> fstdistfunc_;
		size_t data_size_;
		size_t dim_;
	public:
		L2Space(size_t dim) {
			fstdistfunc_ = InnerProduct;
			dim_ = dim;
			data_size_ = dim * sizeof(float);
		}

		size_t get_data_size() {
			return data_size_;
		}
		DISTFUNC<float> get_dist_func() {
			return fstdistfunc_;
		}
		void *get_dist_func_param() {
			return &dim_;
		}

    };

}
