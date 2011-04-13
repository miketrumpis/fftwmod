#ifdef __CPLUSPLUS__
extern "C" {
#endif

#ifndef __GNUC__
#pragma warning(disable: 4275)
#pragma warning(disable: 4101)

#endif
#include "Python.h"
#include "blitz/array.h"
#include "compile.h"
#include "frameobject.h"
#include <complex>
#include <math.h>
#include <string>
#include "scxx/object.h"
#include "scxx/list.h"
#include "scxx/tuple.h"
#include "scxx/dict.h"
#include <iostream>
#include <stdio.h>
#include "numpy/arrayobject.h"
#include <fftw3.h>




// global None value for use in functions.
namespace py {
object None = object(Py_None);
}

const char* find_type(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    if(PyFile_Check(py_obj)) return "file";
    if(PyModule_Check(py_obj)) return "module";

    //should probably do more intergation (and thinking) on these.
    if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance";
    if(PyCallable_Check(py_obj)) return "callable";
    return "unkown type";
}

void throw_error(PyObject* exc, const char* msg)
{
 //printf("setting python error: %s\n",msg);
  PyErr_SetString(exc, msg);
  //printf("throwing error\n");
  throw 1;
}

void handle_bad_type(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}

void handle_conversion_error(PyObject* py_obj, const char* good_type, const char* var_name)
{
    char msg[500];
    sprintf(msg,"Conversion Error:, received '%s' type instead of '%s' for variable '%s'",
            find_type(py_obj),good_type,var_name);
    throw_error(PyExc_TypeError,msg);
}


class int_handler
{
public:
    int convert_to_int(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInt_Check(py_obj))
            handle_conversion_error(py_obj,"int", name);
        return (int) PyInt_AsLong(py_obj);
    }

    int py_to_int(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInt_Check(py_obj))
            handle_bad_type(py_obj,"int", name);
        
        return (int) PyInt_AsLong(py_obj);
    }
};

int_handler x__int_handler = int_handler();
#define convert_to_int(py_obj,name) \
        x__int_handler.convert_to_int(py_obj,name)
#define py_to_int(py_obj,name) \
        x__int_handler.py_to_int(py_obj,name)


PyObject* int_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class float_handler
{
public:
    double convert_to_float(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_conversion_error(py_obj,"float", name);
        return PyFloat_AsDouble(py_obj);
    }

    double py_to_float(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFloat_Check(py_obj))
            handle_bad_type(py_obj,"float", name);
        
        return PyFloat_AsDouble(py_obj);
    }
};

float_handler x__float_handler = float_handler();
#define convert_to_float(py_obj,name) \
        x__float_handler.convert_to_float(py_obj,name)
#define py_to_float(py_obj,name) \
        x__float_handler.py_to_float(py_obj,name)


PyObject* float_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class complex_handler
{
public:
    std::complex<double> convert_to_complex(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_conversion_error(py_obj,"complex", name);
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }

    std::complex<double> py_to_complex(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyComplex_Check(py_obj))
            handle_bad_type(py_obj,"complex", name);
        
        return std::complex<double>(PyComplex_RealAsDouble(py_obj),PyComplex_ImagAsDouble(py_obj));
    }
};

complex_handler x__complex_handler = complex_handler();
#define convert_to_complex(py_obj,name) \
        x__complex_handler.convert_to_complex(py_obj,name)
#define py_to_complex(py_obj,name) \
        x__complex_handler.py_to_complex(py_obj,name)


PyObject* complex_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class unicode_handler
{
public:
    Py_UNICODE* convert_to_unicode(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_conversion_error(py_obj,"unicode", name);
        return PyUnicode_AS_UNICODE(py_obj);
    }

    Py_UNICODE* py_to_unicode(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyUnicode_Check(py_obj))
            handle_bad_type(py_obj,"unicode", name);
        Py_XINCREF(py_obj);
        return PyUnicode_AS_UNICODE(py_obj);
    }
};

unicode_handler x__unicode_handler = unicode_handler();
#define convert_to_unicode(py_obj,name) \
        x__unicode_handler.convert_to_unicode(py_obj,name)
#define py_to_unicode(py_obj,name) \
        x__unicode_handler.py_to_unicode(py_obj,name)


PyObject* unicode_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class string_handler
{
public:
    std::string convert_to_string(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyString_Check(py_obj))
            handle_conversion_error(py_obj,"string", name);
        return std::string(PyString_AsString(py_obj));
    }

    std::string py_to_string(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyString_Check(py_obj))
            handle_bad_type(py_obj,"string", name);
        Py_XINCREF(py_obj);
        return std::string(PyString_AsString(py_obj));
    }
};

string_handler x__string_handler = string_handler();
#define convert_to_string(py_obj,name) \
        x__string_handler.convert_to_string(py_obj,name)
#define py_to_string(py_obj,name) \
        x__string_handler.py_to_string(py_obj,name)


               PyObject* string_to_py(std::string s)
               {
                   return PyString_FromString(s.c_str());
               }
               
class list_handler
{
public:
    py::list convert_to_list(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyList_Check(py_obj))
            handle_conversion_error(py_obj,"list", name);
        return py::list(py_obj);
    }

    py::list py_to_list(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyList_Check(py_obj))
            handle_bad_type(py_obj,"list", name);
        
        return py::list(py_obj);
    }
};

list_handler x__list_handler = list_handler();
#define convert_to_list(py_obj,name) \
        x__list_handler.convert_to_list(py_obj,name)
#define py_to_list(py_obj,name) \
        x__list_handler.py_to_list(py_obj,name)


PyObject* list_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class dict_handler
{
public:
    py::dict convert_to_dict(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyDict_Check(py_obj))
            handle_conversion_error(py_obj,"dict", name);
        return py::dict(py_obj);
    }

    py::dict py_to_dict(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyDict_Check(py_obj))
            handle_bad_type(py_obj,"dict", name);
        
        return py::dict(py_obj);
    }
};

dict_handler x__dict_handler = dict_handler();
#define convert_to_dict(py_obj,name) \
        x__dict_handler.convert_to_dict(py_obj,name)
#define py_to_dict(py_obj,name) \
        x__dict_handler.py_to_dict(py_obj,name)


PyObject* dict_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class tuple_handler
{
public:
    py::tuple convert_to_tuple(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_conversion_error(py_obj,"tuple", name);
        return py::tuple(py_obj);
    }

    py::tuple py_to_tuple(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyTuple_Check(py_obj))
            handle_bad_type(py_obj,"tuple", name);
        
        return py::tuple(py_obj);
    }
};

tuple_handler x__tuple_handler = tuple_handler();
#define convert_to_tuple(py_obj,name) \
        x__tuple_handler.convert_to_tuple(py_obj,name)
#define py_to_tuple(py_obj,name) \
        x__tuple_handler.py_to_tuple(py_obj,name)


PyObject* tuple_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class file_handler
{
public:
    FILE* convert_to_file(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyFile_Check(py_obj))
            handle_conversion_error(py_obj,"file", name);
        return PyFile_AsFile(py_obj);
    }

    FILE* py_to_file(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyFile_Check(py_obj))
            handle_bad_type(py_obj,"file", name);
        Py_XINCREF(py_obj);
        return PyFile_AsFile(py_obj);
    }
};

file_handler x__file_handler = file_handler();
#define convert_to_file(py_obj,name) \
        x__file_handler.convert_to_file(py_obj,name)
#define py_to_file(py_obj,name) \
        x__file_handler.py_to_file(py_obj,name)


               PyObject* file_to_py(FILE* file, const char* name,
                                    const char* mode)
               {
                   return (PyObject*) PyFile_FromFile(file,
                     const_cast<char*>(name),
                     const_cast<char*>(mode), fclose);
               }
               
class instance_handler
{
public:
    py::object convert_to_instance(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_conversion_error(py_obj,"instance", name);
        return py::object(py_obj);
    }

    py::object py_to_instance(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyInstance_Check(py_obj))
            handle_bad_type(py_obj,"instance", name);
        
        return py::object(py_obj);
    }
};

instance_handler x__instance_handler = instance_handler();
#define convert_to_instance(py_obj,name) \
        x__instance_handler.convert_to_instance(py_obj,name)
#define py_to_instance(py_obj,name) \
        x__instance_handler.py_to_instance(py_obj,name)


PyObject* instance_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class numpy_size_handler
{
public:
    void conversion_numpy_check_size(PyArrayObject* arr_obj, int Ndims,
                                     const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"Conversion Error: received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_size(PyArrayObject* arr_obj, int Ndims, const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_size_handler x__numpy_size_handler = numpy_size_handler();
#define conversion_numpy_check_size x__numpy_size_handler.conversion_numpy_check_size
#define numpy_check_size x__numpy_size_handler.numpy_check_size


class numpy_type_handler
{
public:
    void conversion_numpy_check_type(PyArrayObject* arr_obj, int numeric_type,
                                     const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {

        const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                "float", "double", "longdouble", "cfloat", "cdouble",
                                "clongdouble", "object", "string", "unicode", "void", "ntype",
                                "unknown"};
        char msg[500];
        sprintf(msg,"Conversion Error: received '%s' typed array instead of '%s' typed array for variable '%s'",
                type_names[arr_type],type_names[numeric_type],name);
        throw_error(PyExc_TypeError,msg);
        }
    }

    void numpy_check_type(PyArrayObject* arr_obj, int numeric_type, const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if (PyTypeNum_ISEXTENDED(numeric_type))
        {
        char msg[80];
        sprintf(msg, "Conversion Error: extended types not supported for variable '%s'",
                name);
        throw_error(PyExc_TypeError, msg);
        }
        if (!PyArray_EquivTypenums(arr_type, numeric_type))
        {
            const char* type_names[23] = {"bool", "byte", "ubyte","short", "ushort",
                                    "int", "uint", "long", "ulong", "longlong", "ulonglong",
                                    "float", "double", "longdouble", "cfloat", "cdouble",
                                    "clongdouble", "object", "string", "unicode", "void", "ntype",
                                    "unknown"};
            char msg[500];
            sprintf(msg,"received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);
        }
    }
};

numpy_type_handler x__numpy_type_handler = numpy_type_handler();
#define conversion_numpy_check_type x__numpy_type_handler.conversion_numpy_check_type
#define numpy_check_type x__numpy_type_handler.numpy_check_type


class numpy_handler
{
public:
    PyArrayObject* convert_to_numpy(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        Py_XINCREF(py_obj);
        if (!py_obj || !PyArray_Check(py_obj))
            handle_conversion_error(py_obj,"numpy", name);
        return (PyArrayObject*) py_obj;
    }

    PyArrayObject* py_to_numpy(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !PyArray_Check(py_obj))
            handle_bad_type(py_obj,"numpy", name);
        Py_XINCREF(py_obj);
        return (PyArrayObject*) py_obj;
    }
};

numpy_handler x__numpy_handler = numpy_handler();
#define convert_to_numpy(py_obj,name) \
        x__numpy_handler.convert_to_numpy(py_obj,name)
#define py_to_numpy(py_obj,name) \
        x__numpy_handler.py_to_numpy(py_obj,name)


PyObject* numpy_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}


class catchall_handler
{
public:
    py::object convert_to_catchall(PyObject* py_obj, const char* name)
    {
        // Incref occurs even if conversion fails so that
        // the decref in cleanup_code has a matching incref.
        
        if (!py_obj || !(py_obj))
            handle_conversion_error(py_obj,"catchall", name);
        return py::object(py_obj);
    }

    py::object py_to_catchall(PyObject* py_obj, const char* name)
    {
        // !! Pretty sure INCREF should only be called on success since
        // !! py_to_xxx is used by the user -- not the code generator.
        if (!py_obj || !(py_obj))
            handle_bad_type(py_obj,"catchall", name);
        
        return py::object(py_obj);
    }
};

catchall_handler x__catchall_handler = catchall_handler();
#define convert_to_catchall(py_obj,name) \
        x__catchall_handler.convert_to_catchall(py_obj,name)
#define py_to_catchall(py_obj,name) \
        x__catchall_handler.py_to_catchall(py_obj,name)


PyObject* catchall_to_py(PyObject* obj)
{
    return (PyObject*) obj;
}

#include <fftw3.h>
#include <list>
#include <iostream>
#ifdef THREADED
#include <pthread.h>
#endif
#ifndef NTHREADS
#define NTHREADS 1
#endif

#define INVERSE +1
#define FORWARD -1

#define MAX(a,b) ( (a) > (b) ? (a) : (b) );

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;
  
typedef struct {
  char *i;
  char *o;
  char *p;
  fftw_iodim *ft_dims;
  int ft_rank;
  fftw_iodim *nft_dims;
  int nft_rank;
  int shift;
  int direction;
} fft_args;

inline int idot(int *v1, int *v2, int N)
{
  /* Return the dot product of v1 and v2 */
  int k, p=0;
  for(k=0; k<N; k++) p += v1[k]*v2[k];
  return p;
}

inline 
int isum(int *v1, int N)
{
  /* Return the sum of the N entries in v1 */
  int k, p=0;
  for(k=0; k<N; k++) p += v1[k];
  return p;
}

inline
void indices_int(int L, int *shape, int *idc, int N)
{
  /* Given a flat index L into a ND grid with dims listed in shape,
     return the grid coordinates in idc
  */
  int i, nn=L;
  for(i=0; i<N; i++) {
    idc[N-i-1] = nn % shape[N-i-1];
    nn /= shape[N-i-1];
  }
}

void *thread_fftwf(void *ptr)
{
  fft_args *args = (fft_args *) ptr;
  fftwf_plan *FT = (fftwf_plan *) args->p;
  fftwf_complex *z_i, *z_o, *o_ptr, *i_ptr;
  float oscl, mod, shift_fix, len_xform = 1.0;
  int k, l, offsets, nxforms = 1;
  int ft_rank = args->ft_rank, nft_rank = args->nft_rank;
  int *ft_shape, *ft_strides, *ft_indices, *oft_strides;
  int *nft_shape, *nft_strides, *nft_indices, *onft_strides;

  ft_shape = new int[ft_rank];
  ft_strides = new int[ft_rank];
  ft_indices = new int[ft_rank];
  oft_strides = new int[ft_rank];

  nft_shape = new int[nft_rank];
  nft_strides = new int[nft_rank];
  nft_indices = new int[nft_rank];
  onft_strides = new int[nft_rank];

  for(k=0; k<ft_rank; k++) {
    len_xform *= args->ft_dims[k].n;
    ft_shape[k] = args->ft_dims[k].n;
    ft_strides[k] = args->ft_dims[k].is;
    oft_strides[k] = args->ft_dims[k].os;
  }
  for(k=0; k<nft_rank; k++) {
    nxforms *= args->nft_dims[k].n;
    nft_shape[k] = args->nft_dims[k].n;
    nft_strides[k] = args->nft_dims[k].is;
    onft_strides[k] = args->nft_dims[k].os;
  }
  z_i = reinterpret_cast<fftwf_complex*>( args->i );
  z_o = reinterpret_cast<fftwf_complex*>( args->o );

  // If shift is true, then len(dim_i)/2 is treated as x_i = 0,
  // and each axis grid is x_i(n) = n - len(dim_i)/2 = n + i0.
  // Therefore the term (-1)**(x_i+x_j+x_k+...) can be computed
  // as (-1)**(i+j+k+...) * (-1)**(i0+j0+k0+...)
  offsets = 0;
  for(k=0; k<ft_rank; k++) offsets += ft_shape[k]/2;
  shift_fix = offsets % 2 ? -1.0 : 1.0;

  if(args->shift) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
// 	mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
    }
  }
  fftwf_execute_dft(*FT, z_i, z_o);

  if(args->shift) {
    // if input and output pointers are different, demodulate
    // input separately without scaling on the IFFT
//     oscl = (args->direction==INVERSE) ? 1.0/len_xform : 1.0;
    oscl = (args->direction==INVERSE) ? shift_fix/len_xform : shift_fix;
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      if(args->i != args->o) {
  	o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
	for(l=0; l<(int)len_xform; l++) {
  	  indices_int(l, ft_shape, ft_indices, ft_rank);
// 	  mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	  mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	  i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
	  (*i_ptr)[0] *= mod;
	  (*i_ptr)[1] *= mod;
	}
      }
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
	mod = isum(ft_indices, ft_rank)%2 ? -oscl : oscl;
  	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
				      
    }
  } else if(args->direction == INVERSE) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
 	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] /= len_xform;
 	(*i_ptr)[1] /= len_xform;
      }
    }
  }
    


  return NULL;
}

template <const int N_rank>
void cfloat_fft(blitz::Array<cfloat,N_rank>& ai, 
		blitz::Array<cfloat,N_rank>& ao,
		int fft_rank, int *fft_dims, int direction, int shift)
{
  int n, k;
  for(k=0; k<fft_rank; k++) {
    while(fft_dims[k] < 0) fft_dims[k] += N_rank;
    if(fft_dims[k] >= N_rank) {
      std::cout<<"fft_dim is not within the rank of the array: "<<fft_dims[k]<<std::endl;
      return;
    }
  }

  fftwf_plan FT;
  // howmany_rank -- make sure it is at least 1 (with a degenerate dim if necessary)
  int howmany_rank = MAX(1, ai.rank() - fft_rank); 
  int plan_flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
  int n_threads;
  fftw_iodim *fft_iodims = new fftw_iodim[fft_rank];
  fftw_iodim *howmany_dims = new fftw_iodim[howmany_rank];
  // put down reasonable defaults for 1st non-FFT dimension
  howmany_dims[0].n = 1; howmany_dims[0].is = 1; howmany_dims[0].os = 1;
  // the fft_dims are specified in the arguments, 
  // so they are straightforward to put down
  for(k=0; k<fft_rank; k++) {
    fft_iodims[k].n = ai.shape()[ fft_dims[k] ];
    fft_iodims[k].is = ai.stride()[ fft_dims[k] ];
    fft_iodims[k].os = ao.stride()[ fft_dims[k] ];
  }
  // the "other" dims are a little more tricky..
  // 1) make a list of all the dims
  std::list<int> a_dims;
  std::list<int>::iterator it;
  for(k=0; k<ai.rank(); k++)
    a_dims.push_back(k);
  // 2) prune the fft_dims from the list
  for(k=0; k<fft_rank; k++)
    a_dims.remove(fft_dims[k]);
  // 3) put down the remaining dim info in howmany_dims
  n = 0; //howmany_rank-1;
  for(it = a_dims.begin(); it != a_dims.end(); it++) {
    howmany_dims[n].n = ai.shape()[ *it ];
    howmany_dims[n].is = ai.stride()[ *it ];
    howmany_dims[n].os = ao.stride()[ *it ];
    n++;
  }

#ifdef THREADED
  // short circuit eval I HOPE!
  // Do threads IF there are multiple FFTs AND the length of the 
  // leading non-FFT dimension is divided by NTHREADS w/o remainder
  pthread_t threads[NTHREADS];
  if( howmany_rank && !(howmany_dims[0].n % NTHREADS) ) {
    n_threads = NTHREADS;
  } else {
#endif
    n_threads = 1;
#ifdef THREADED
  }
#endif
  fft_args *args = new fft_args[n_threads];
  int block_sz = howmany_dims[0].n / n_threads;
  howmany_dims[0].n = block_sz;
  FT = fftwf_plan_guru_dft(fft_rank, fft_iodims, howmany_rank, howmany_dims,
			   reinterpret_cast<fftwf_complex*>( ai.data() ),
			   reinterpret_cast<fftwf_complex*>( ao.data() ),
			   direction, plan_flags);
  if(FT==NULL) {
    std::cout << "FFTW created a null plan, exiting" << std::endl;
    return;
  }
  

  for(n=0; n<n_threads; n++) {
    (args+n)->i = (char *) (ai.dataZero() + n*block_sz*howmany_dims[0].is);
    (args+n)->o = (char *) (ao.dataZero() + n*block_sz*howmany_dims[0].os);
    (args+n)->p = (char *) &FT;
    (args+n)->ft_dims = fft_iodims;
    (args+n)->ft_rank = fft_rank;
    (args+n)->nft_dims = howmany_dims;
    (args+n)->nft_rank = howmany_rank;
    (args+n)->shift = shift;
    (args+n)->direction = direction;
#ifdef THREADED
    pthread_create(&(threads[n]), NULL, thread_fftwf, (void *) (args+n));
#endif
  }
#ifdef THREADED
  for(n=0; n<n_threads; n++) {
    pthread_join(threads[n], NULL);
  }
#else
  void *throw_away;
  throw_away = thread_fftwf( (void *) args);
#endif
  fftwf_destroy_plan(FT);
  delete [] args;
}

void *thread_fftw(void *ptr)
{
  fft_args *args = (fft_args *) ptr;
  fftw_plan *FT = (fftw_plan *) args->p;
  fftw_complex *z_i, *z_o, *o_ptr, *i_ptr;
  double oscl, mod, shift_fix, len_xform = 1.0;
  int k, l, offsets, nxforms = 1;
  int ft_rank = args->ft_rank, nft_rank = args->nft_rank;
  int *ft_shape, *ft_strides, *ft_indices, *oft_strides;
  int *nft_shape, *nft_strides, *nft_indices, *onft_strides;

  ft_shape = new int[ft_rank];
  ft_strides = new int[ft_rank];
  ft_indices = new int[ft_rank];
  oft_strides = new int[ft_rank];

  nft_shape = new int[nft_rank];
  nft_strides = new int[nft_rank];
  nft_indices = new int[nft_rank];
  onft_strides = new int[nft_rank];

  for(k=0; k<ft_rank; k++) {
    len_xform *= args->ft_dims[k].n;
    ft_shape[k] = args->ft_dims[k].n;
    ft_strides[k] = args->ft_dims[k].is;
    oft_strides[k] = args->ft_dims[k].os;
  }
  for(k=0; k<nft_rank; k++) {
    nxforms *= args->nft_dims[k].n;
    nft_shape[k] = args->nft_dims[k].n;
    nft_strides[k] = args->nft_dims[k].is;
    onft_strides[k] = args->nft_dims[k].os;
  }
  z_i = reinterpret_cast<fftw_complex*>( args->i );
  z_o = reinterpret_cast<fftw_complex*>( args->o );

  // If shift is true, then len(dim_i)/2 is treated as x_i = 0,
  // and each axis grid is x_i(n) = n - len(dim_i)/2 = n + i0.
  // Therefore the term (-1)**(x_i+x_j+x_k+...) can be computed
  // as (-1)**(i+j+k+...) * (-1)**(i0+j0+k0+...)
  offsets = 0;
  for(k=0; k<ft_rank; k++) offsets += ft_shape[k]/2;
  shift_fix = offsets % 2 ? -1.0 : 1.0;

  if(args->shift) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
// 	mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
    }
  }
  fftw_execute_dft(*FT, z_i, z_o);

  if(args->shift) {
    // if input and output pointers are different, demodulate
    // input separately without scaling on the IFFT
//     oscl = (args->direction==INVERSE) ? 1.0/len_xform : 1.0;
    oscl = (args->direction==INVERSE) ? shift_fix/len_xform : shift_fix;
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      if(args->i != args->o) {
  	o_ptr = z_i + idot(nft_indices, nft_strides, nft_rank);
	for(l=0; l<(int)len_xform; l++) {
  	  indices_int(l, ft_shape, ft_indices, ft_rank);
// 	  mod = isum(ft_indices, ft_rank)%2 ? -1.0 : 1.0;
	  mod = isum(ft_indices, ft_rank)%2 ? -shift_fix : shift_fix;
  	  i_ptr = o_ptr + idot(ft_indices, ft_strides, ft_rank);
	  (*i_ptr)[0] *= mod;
	  (*i_ptr)[1] *= mod;
	}
      }
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
	mod = isum(ft_indices, ft_rank)%2 ? -oscl : oscl;
  	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] *= mod;
 	(*i_ptr)[1] *= mod;
      }
				      
    }
  } else if(args->direction == INVERSE) {
    for(k=0; k<nxforms; k++) {
      indices_int(k, nft_shape, nft_indices, nft_rank);
      o_ptr = z_o + idot(nft_indices, onft_strides, nft_rank);
      for(l=0; l<(int)len_xform; l++) {
  	indices_int(l, ft_shape, ft_indices, ft_rank);
 	i_ptr = o_ptr + idot(ft_indices, oft_strides, ft_rank);
 	(*i_ptr)[0] /= len_xform;
 	(*i_ptr)[1] /= len_xform;
      }
    }
  }
    


  return NULL;
}

template <const int N_rank>
void cdouble_fft(blitz::Array<cdouble,N_rank>& ai, 
		 blitz::Array<cdouble,N_rank>& ao,
		 int fft_rank, int *fft_dims, int direction, int shift)
{
  int n, k;
  for(k=0; k<fft_rank; k++) {
    while(fft_dims[k] < 0) fft_dims[k] += N_rank;
    if(fft_dims[k] >= N_rank) {
      std::cout<<"fft_dim is not within the rank of the array: "<<fft_dims[k]<<std::endl;
      return;
    }
  }

  fftw_plan FT;
  // howmany_rank -- make sure it is at least 1 (with a degenerate dim if necessary)
  int howmany_rank = MAX(1, ai.rank() - fft_rank);
  int plan_flags = FFTW_ESTIMATE | FFTW_PRESERVE_INPUT;
  int n_threads;
  fftw_iodim *fft_iodims = new fftw_iodim[fft_rank];
  fftw_iodim *howmany_dims = new fftw_iodim[howmany_rank];
  // put down reasonable defaults for 1st non-FFT dimension
  howmany_dims[0].n = 1; howmany_dims[0].is = 1; howmany_dims[0].os = 1;
  // the fft_dims are specified in the arguments, 
  // so they are straightforward to put down
  for(k=0; k<fft_rank; k++) {
    fft_iodims[k].n = ai.shape()[ fft_dims[k] ];
    fft_iodims[k].is = ai.stride()[ fft_dims[k] ];
    fft_iodims[k].os = ao.stride()[ fft_dims[k] ];
  }
  // the "other" dims are a little more tricky..
  // 1) make a list of all the dims
  std::list<int> a_dims;
  std::list<int>::iterator it;
  for(k=0; k<ai.rank(); k++)
    a_dims.push_back(k);
  // 2) prune the fft_dims from the list
  for(k=0; k<fft_rank; k++)
    a_dims.remove(fft_dims[k]);
  // 3) put down the remaining dim info in howmany_dims
  n = 0; //howmany_rank-1;
  for(it = a_dims.begin(); it != a_dims.end(); it++) {
    howmany_dims[n].n = ai.shape()[ *it ];
    howmany_dims[n].is = ai.stride()[ *it ];
    howmany_dims[n].os = ao.stride()[ *it ];
    n++;
  }
  
#ifdef THREADED
  // short circuit eval I HOPE!
  // Do threads IF there are multiple FFTs AND the length of the 
  // leading non-FFT dimension is divided by NTHREADS w/o remainder
  pthread_t threads[NTHREADS];
  if( howmany_rank && !(howmany_dims[0].n % NTHREADS) ) {
    n_threads = NTHREADS;
  } else {
#endif
    n_threads = 1;
#ifdef THREADED
  }
#endif
  fft_args *args = new fft_args[n_threads];
  int block_sz = howmany_dims[0].n / n_threads;
  howmany_dims[0].n = block_sz;
  FT = fftw_plan_guru_dft(fft_rank, fft_iodims, howmany_rank, howmany_dims,
			  reinterpret_cast<fftw_complex*>( ai.data() ),
			  reinterpret_cast<fftw_complex*>( ao.data() ),
			  direction, plan_flags);
  if(FT==NULL) {
    std::cout << "FFTW created a null plan, exiting" << std::endl;
    return;
  }
  
  for(n=0; n<n_threads; n++) {
    (args+n)->i = (char *) (ai.dataZero() + n*block_sz*howmany_dims[0].is);
    (args+n)->o = (char *) (ao.dataZero() + n*block_sz*howmany_dims[0].os);
    (args+n)->p = (char *) &FT;
    (args+n)->ft_dims = fft_iodims;
    (args+n)->ft_rank = fft_rank;
    (args+n)->nft_dims = howmany_dims;
    (args+n)->nft_rank = howmany_rank;
    (args+n)->shift = shift;
    (args+n)->direction = direction;
#ifdef THREADED
    pthread_create(&(threads[n]), NULL, thread_fftw, (void *) (args+n));
#endif
  }
#ifdef THREADED
  for(n=0; n<n_threads; n++) {
    pthread_join(threads[n], NULL);
  }
#else
  void *throw_away;
  throw_away = thread_fftw( (void *) args);
#endif
  fftw_destroy_plan(FT);
  delete [] args;
}



// This should be declared only if they are used by some function
// to keep from generating needless warnings. for now, we'll always
// declare them.

int _beg = blitz::fromStart;
int _end = blitz::toEnd;
blitz::Range _all = blitz::Range::all();

template<class T, int N>
static blitz::Array<T,N> convert_to_blitz(PyArrayObject* arr_obj,const char* name)
{
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyArrayObject* arr_obj,const char* name)
{

    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}


static PyObject* _fft_D_1(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_1",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,1,"a");
        blitz::Array<std::complex<double> ,1> a = convert_to_blitz<std::complex<double> ,1>(a_array,"a");
        blitz::TinyVector<int,1> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,1,"b");
        blitz::Array<std::complex<double> ,1> b = convert_to_blitz<std::complex<double> ,1>(b_array,"b");
        blitz::TinyVector<int,1> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_2(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_2",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,2,"a");
        blitz::Array<std::complex<double> ,2> a = convert_to_blitz<std::complex<double> ,2>(a_array,"a");
        blitz::TinyVector<int,2> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,2,"b");
        blitz::Array<std::complex<double> ,2> b = convert_to_blitz<std::complex<double> ,2>(b_array,"b");
        blitz::TinyVector<int,2> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_3(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_3",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,3,"a");
        blitz::Array<std::complex<double> ,3> a = convert_to_blitz<std::complex<double> ,3>(a_array,"a");
        blitz::TinyVector<int,3> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,3,"b");
        blitz::Array<std::complex<double> ,3> b = convert_to_blitz<std::complex<double> ,3>(b_array,"b");
        blitz::TinyVector<int,3> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_4(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_4",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,4,"a");
        blitz::Array<std::complex<double> ,4> a = convert_to_blitz<std::complex<double> ,4>(a_array,"a");
        blitz::TinyVector<int,4> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,4,"b");
        blitz::Array<std::complex<double> ,4> b = convert_to_blitz<std::complex<double> ,4>(b_array,"b");
        blitz::TinyVector<int,4> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_5(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_5",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,5,"a");
        blitz::Array<std::complex<double> ,5> a = convert_to_blitz<std::complex<double> ,5>(a_array,"a");
        blitz::TinyVector<int,5> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,5,"b");
        blitz::Array<std::complex<double> ,5> b = convert_to_blitz<std::complex<double> ,5>(b_array,"b");
        blitz::TinyVector<int,5> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_6(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_6",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,6,"a");
        blitz::Array<std::complex<double> ,6> a = convert_to_blitz<std::complex<double> ,6>(a_array,"a");
        blitz::TinyVector<int,6> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,6,"b");
        blitz::Array<std::complex<double> ,6> b = convert_to_blitz<std::complex<double> ,6>(b_array,"b");
        blitz::TinyVector<int,6> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_7(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_7",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,7,"a");
        blitz::Array<std::complex<double> ,7> a = convert_to_blitz<std::complex<double> ,7>(a_array,"a");
        blitz::TinyVector<int,7> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,7,"b");
        blitz::Array<std::complex<double> ,7> b = convert_to_blitz<std::complex<double> ,7>(b_array,"b");
        blitz::TinyVector<int,7> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_8(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_8",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,8,"a");
        blitz::Array<std::complex<double> ,8> a = convert_to_blitz<std::complex<double> ,8>(a_array,"a");
        blitz::TinyVector<int,8> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,8,"b");
        blitz::Array<std::complex<double> ,8> b = convert_to_blitz<std::complex<double> ,8>(b_array,"b");
        blitz::TinyVector<int,8> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_9(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_9",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,9,"a");
        blitz::Array<std::complex<double> ,9> a = convert_to_blitz<std::complex<double> ,9>(a_array,"a");
        blitz::TinyVector<int,9> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,9,"b");
        blitz::Array<std::complex<double> ,9> b = convert_to_blitz<std::complex<double> ,9>(b_array,"b");
        blitz::TinyVector<int,9> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_10(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_10",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,10,"a");
        blitz::Array<std::complex<double> ,10> a = convert_to_blitz<std::complex<double> ,10>(a_array,"a");
        blitz::TinyVector<int,10> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,10,"b");
        blitz::Array<std::complex<double> ,10> b = convert_to_blitz<std::complex<double> ,10>(b_array,"b");
        blitz::TinyVector<int,10> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_D_11(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_D_11",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CDOUBLE,"a");
        conversion_numpy_check_size(a_array,11,"a");
        blitz::Array<std::complex<double> ,11> a = convert_to_blitz<std::complex<double> ,11>(a_array,"a");
        blitz::TinyVector<int,11> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CDOUBLE,"b");
        conversion_numpy_check_size(b_array,11,"b");
        blitz::Array<std::complex<double> ,11> b = convert_to_blitz<std::complex<double> ,11>(b_array,"b");
        blitz::TinyVector<int,11> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cdouble_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cdouble_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_1(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_1",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,1,"a");
        blitz::Array<std::complex<float> ,1> a = convert_to_blitz<std::complex<float> ,1>(a_array,"a");
        blitz::TinyVector<int,1> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,1,"b");
        blitz::Array<std::complex<float> ,1> b = convert_to_blitz<std::complex<float> ,1>(b_array,"b");
        blitz::TinyVector<int,1> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_2(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_2",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,2,"a");
        blitz::Array<std::complex<float> ,2> a = convert_to_blitz<std::complex<float> ,2>(a_array,"a");
        blitz::TinyVector<int,2> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,2,"b");
        blitz::Array<std::complex<float> ,2> b = convert_to_blitz<std::complex<float> ,2>(b_array,"b");
        blitz::TinyVector<int,2> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_3(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_3",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,3,"a");
        blitz::Array<std::complex<float> ,3> a = convert_to_blitz<std::complex<float> ,3>(a_array,"a");
        blitz::TinyVector<int,3> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,3,"b");
        blitz::Array<std::complex<float> ,3> b = convert_to_blitz<std::complex<float> ,3>(b_array,"b");
        blitz::TinyVector<int,3> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_4(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_4",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,4,"a");
        blitz::Array<std::complex<float> ,4> a = convert_to_blitz<std::complex<float> ,4>(a_array,"a");
        blitz::TinyVector<int,4> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,4,"b");
        blitz::Array<std::complex<float> ,4> b = convert_to_blitz<std::complex<float> ,4>(b_array,"b");
        blitz::TinyVector<int,4> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_5(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_5",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,5,"a");
        blitz::Array<std::complex<float> ,5> a = convert_to_blitz<std::complex<float> ,5>(a_array,"a");
        blitz::TinyVector<int,5> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,5,"b");
        blitz::Array<std::complex<float> ,5> b = convert_to_blitz<std::complex<float> ,5>(b_array,"b");
        blitz::TinyVector<int,5> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_6(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_6",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,6,"a");
        blitz::Array<std::complex<float> ,6> a = convert_to_blitz<std::complex<float> ,6>(a_array,"a");
        blitz::TinyVector<int,6> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,6,"b");
        blitz::Array<std::complex<float> ,6> b = convert_to_blitz<std::complex<float> ,6>(b_array,"b");
        blitz::TinyVector<int,6> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_7(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_7",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,7,"a");
        blitz::Array<std::complex<float> ,7> a = convert_to_blitz<std::complex<float> ,7>(a_array,"a");
        blitz::TinyVector<int,7> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,7,"b");
        blitz::Array<std::complex<float> ,7> b = convert_to_blitz<std::complex<float> ,7>(b_array,"b");
        blitz::TinyVector<int,7> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_8(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_8",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,8,"a");
        blitz::Array<std::complex<float> ,8> a = convert_to_blitz<std::complex<float> ,8>(a_array,"a");
        blitz::TinyVector<int,8> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,8,"b");
        blitz::Array<std::complex<float> ,8> b = convert_to_blitz<std::complex<float> ,8>(b_array,"b");
        blitz::TinyVector<int,8> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_9(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_9",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,9,"a");
        blitz::Array<std::complex<float> ,9> a = convert_to_blitz<std::complex<float> ,9>(a_array,"a");
        blitz::TinyVector<int,9> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,9,"b");
        blitz::Array<std::complex<float> ,9> b = convert_to_blitz<std::complex<float> ,9>(b_array,"b");
        blitz::TinyVector<int,9> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_10(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_10",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,10,"a");
        blitz::Array<std::complex<float> ,10> a = convert_to_blitz<std::complex<float> ,10>(a_array,"a");
        blitz::TinyVector<int,10> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,10,"b");
        blitz::Array<std::complex<float> ,10> b = convert_to_blitz<std::complex<float> ,10>(b_array,"b");
        blitz::TinyVector<int,10> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                
static PyObject* _fft_F_11(PyObject*self, PyObject* args, PyObject* kywds)
{
    py::object return_val;
    int exception_occured = 0;
    PyObject *py_local_dict = NULL;
    static const char *kwlist[] = {"a","b","adims","fft_sign","shift","inplace","local_dict", NULL};
    PyObject *py_a, *py_b, *py_adims, *py_fft_sign, *py_shift, *py_inplace;
    int a_used, b_used, adims_used, fft_sign_used, shift_used, inplace_used;
    py_a = py_b = py_adims = py_fft_sign = py_shift = py_inplace = NULL;
    a_used= b_used= adims_used= fft_sign_used= shift_used= inplace_used = 0;
    
    if(!PyArg_ParseTupleAndKeywords(args,kywds,"OOOOOO|O:_fft_F_11",const_cast<char**>(kwlist),&py_a, &py_b, &py_adims, &py_fft_sign, &py_shift, &py_inplace, &py_local_dict))
       return NULL;
    try                              
    {                                
        py_a = py_a;
        PyArrayObject* a_array = convert_to_numpy(py_a,"a");
        conversion_numpy_check_type(a_array,PyArray_CFLOAT,"a");
        conversion_numpy_check_size(a_array,11,"a");
        blitz::Array<std::complex<float> ,11> a = convert_to_blitz<std::complex<float> ,11>(a_array,"a");
        blitz::TinyVector<int,11> Na = a.shape();
        a_used = 1;
        py_b = py_b;
        PyArrayObject* b_array = convert_to_numpy(py_b,"b");
        conversion_numpy_check_type(b_array,PyArray_CFLOAT,"b");
        conversion_numpy_check_size(b_array,11,"b");
        blitz::Array<std::complex<float> ,11> b = convert_to_blitz<std::complex<float> ,11>(b_array,"b");
        blitz::TinyVector<int,11> Nb = b.shape();
        b_used = 1;
        py_adims = py_adims;
        PyArrayObject* adims_array = convert_to_numpy(py_adims,"adims");
        conversion_numpy_check_type(adims_array,PyArray_INT,"adims");
        conversion_numpy_check_size(adims_array,1,"adims");
        blitz::Array<int,1> adims = convert_to_blitz<int,1>(adims_array,"adims");
        blitz::TinyVector<int,1> Nadims = adims.shape();
        adims_used = 1;
        py_fft_sign = py_fft_sign;
        int fft_sign = convert_to_int(py_fft_sign,"fft_sign");
        fft_sign_used = 1;
        py_shift = py_shift;
        int shift = convert_to_int(py_shift,"shift");
        shift_used = 1;
        py_inplace = py_inplace;
        int inplace = convert_to_int(py_inplace,"inplace");
        inplace_used = 1;
        /*<function call here>*/     
        
                if(inplace) {
                  cfloat_fft(a, a, adims.shape()[0], adims.data(), fft_sign, shift);
                } else {
                  cfloat_fft(a, b, adims.shape()[0], adims.data(), fft_sign, shift);
                }
        if(py_local_dict)                                  
        {                                                  
            py::dict local_dict = py::dict(py_local_dict); 
        }                                                  
    
    }                                
    catch(...)                       
    {                                
        return_val =  py::object();      
        exception_occured = 1;       
    }                                
    /*cleanup code*/                     
    if(a_used)
    {
        Py_XDECREF(py_a);
    }
    if(b_used)
    {
        Py_XDECREF(py_b);
    }
    if(adims_used)
    {
        Py_XDECREF(py_adims);
    }
    if(!(PyObject*)return_val && !exception_occured)
    {
                                  
        return_val = Py_None;            
    }
                                  
    return return_val.disown();           
}                                


static PyMethodDef compiled_methods[] = 
{
    {"_fft_D_1",(PyCFunction)_fft_D_1 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_2",(PyCFunction)_fft_D_2 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_3",(PyCFunction)_fft_D_3 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_4",(PyCFunction)_fft_D_4 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_5",(PyCFunction)_fft_D_5 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_6",(PyCFunction)_fft_D_6 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_7",(PyCFunction)_fft_D_7 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_8",(PyCFunction)_fft_D_8 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_9",(PyCFunction)_fft_D_9 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_10",(PyCFunction)_fft_D_10 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_D_11",(PyCFunction)_fft_D_11 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_1",(PyCFunction)_fft_F_1 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_2",(PyCFunction)_fft_F_2 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_3",(PyCFunction)_fft_F_3 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_4",(PyCFunction)_fft_F_4 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_5",(PyCFunction)_fft_F_5 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_6",(PyCFunction)_fft_F_6 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_7",(PyCFunction)_fft_F_7 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_8",(PyCFunction)_fft_F_8 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_9",(PyCFunction)_fft_F_9 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_10",(PyCFunction)_fft_F_10 , METH_VARARGS|METH_KEYWORDS},
    {"_fft_F_11",(PyCFunction)_fft_F_11 , METH_VARARGS|METH_KEYWORDS},
    {NULL,      NULL}        /* Sentinel */
};

PyMODINIT_FUNC initfft_ext(void)
{
    
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
    (void) Py_InitModule("fft_ext", compiled_methods);
}

#ifdef __CPLUSCPLUS__
}
#endif
