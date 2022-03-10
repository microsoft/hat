import ctypes
import os
import pathlib
import sys
import numpy as np
from functools import reduce
from typing import Dict, List

# lifted from https://github.com/jatinx/PyHIP

# RTC and HIPRT are both served from the same library, but it's subject to change
_libhip_libname = 'libamdhip64.so'

_libhiprtc = _libhip = None
if 'linux' in sys.platform:
    _libhiprtc = _libhip = ctypes.cdll.LoadLibrary(_libhip_libname)
else:
    # TODO
    raise RuntimeError('Only linux is supported')


def POINTER(obj):
    """
    ctype pointer to object
    """
    p = ctypes.POINTER(obj)
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p


_libhiprtc.hiprtcGetErrorString.restype = ctypes.c_char_p
_libhiprtc.hiprtcGetErrorString.argtypes = [ctypes.c_int]


def hiprtcGetErrorString(e):
    """
    Retrieve hiprtc error string.
    Return the string associated with the specified hiprtc error status
    code.
    Parameters
    ----------
    e : int
        Error number.
    Returns
    -------
    s : str
        Error string.
    """

    return _libhiprtc.hiprtcGetErrorString(e)

# Generic hiprtc Error


class hiprtcError(Exception):
    """hiprtc error."""
    pass


class hiprtcErrorOutOfMemory(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(1)
    pass


class hiprtcErrorProgramCreationFailure(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(2)
    pass


class hiprtcErrorInvalidInput(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(3)
    pass


class hiprtcErrorInvalidProgram(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(4)
    pass


class hiprtcErrorInvalidOption(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(5)
    pass


class hiprtcErrorCompilation(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(6)
    pass


class hiprtcErrorBuiltinOperationFailure(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(7)
    pass


class hiprtcErrorNoNameExpressionAfterCompilation(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(8)
    pass


class hiprtcErrorNoLoweredNamesBeforeCompilation(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(9)
    pass


class hiprtcErrorNameExpressionNotValid(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(10)
    pass


class hiprtcErrorInternalError(hiprtcError):
    __doc__ = _libhiprtc.hiprtcGetErrorString(11)
    pass


hiprtcExceptions = {
    1: hiprtcErrorOutOfMemory,
    2: hiprtcErrorProgramCreationFailure,
    3: hiprtcErrorInvalidInput,
    4: hiprtcErrorInvalidProgram,
    5: hiprtcErrorInvalidOption,
    6: hiprtcErrorCompilation,
    7: hiprtcErrorBuiltinOperationFailure,
    8: hiprtcErrorNoNameExpressionAfterCompilation,
    9: hiprtcErrorNoLoweredNamesBeforeCompilation,
    10: hiprtcErrorNameExpressionNotValid,
    11: hiprtcErrorInternalError
}


def hiprtcCheckStatus(status):
    if status != 0:
        try:
            e = hiprtcExceptions[status]
        except KeyError:
            raise hiprtcError('unknown hiprtc error %s' % status)
        else:
            raise e


_libhiprtc.hiprtcCreateProgram.restype = int
_libhiprtc.hiprtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # hiprtcProgram
                                           ctypes.POINTER(
                                               ctypes.c_char),   # Source
                                           ctypes.POINTER(
                                               ctypes.c_char),   # Name
                                           ctypes.c_int,                    # numberOfHeaders
                                           ctypes.POINTER(
                                               ctypes.c_char_p),  # header
                                           ctypes.POINTER(ctypes.c_char_p)]  # headerNames


def hiprtcCreateProgram(source, name, header_names, header_sources):
    """
    Create hiprtcProgram
    Parameters
    ----------
    source : string
        Source in python string
    name : string
        Program name
    header_names: list of string
        list of headernames
    header_sources: list of string
        list of headernames

    Returns
    -------
    prog : ctypes pointer
        hiprtc program handle
    """

    # Encode strings to utf-8
    e_source = source.encode('utf-8')
    e_name = name.encode('utf-8')
    e_header_names = list()
    e_header_sources = list()
    for header_name in header_names:
        e_header_name = header_name.encode('utf-8')
        e_header_names.append(e_header_name)
    for header_source in header_sources:
        e_header_source = header_source.encode('utf-8')
        e_header_sources.append(e_header_source)

    prog = ctypes.c_void_p()
    c_header_names = (ctypes.c_char_p * len(e_header_names))()
    c_header_names[:] = e_header_names
    c_header_sources = (ctypes.c_char_p * len(e_header_sources))()
    c_header_sources[:] = e_header_sources

    status = _libhiprtc.hiprtcCreateProgram(ctypes.byref(
        prog), e_source, e_name, len(e_header_names), c_header_sources, c_header_names)
    hiprtcCheckStatus(status)
    return prog


_libhiprtc.hiprtcDestroyProgram.restype = int
_libhiprtc.hiprtcDestroyProgram.argtypes = [
    ctypes.POINTER(ctypes.c_void_p)]  # hiprtcProgram


def hiprtcDestroyProgram(prog):
    """
    Destroy hiprtcProgram
    Parameters
    ----------
    prog : ctypes pointer
        hiprtc program handle
    """
    status = _libhiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
    hiprtcCheckStatus(status)


_libhiprtc.hiprtcAddNameExpression.restype = int
_libhiprtc.hiprtcAddNameExpression.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                               ctypes.POINTER(ctypes.c_char)]   # expression


def hiprtcAddNameExpression(prog, expression):
    """
    Tracks expression name through hiprtc compilation unit
    Parameters
    ----------
    prog : ctypes pointer
        hiprtc program handle
    expression : string
        exression name
    """
    e_expression = expression.encode('utf-8')
    status = _libhiprtc.hiprtcAddNameExpression(prog, e_expression)
    hiprtcCheckStatus(status)


_libhiprtc.hiprtcCompileProgram.restype = int
_libhiprtc.hiprtcCompileProgram.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                            ctypes.c_int,                    # num of options
                                            ctypes.POINTER(ctypes.c_char_p)]  # options


def hiprtcCompileProgram(prog, options):
    """
    Compiles the hiprtc program
    Parameters
    ----------
    prog : ctypes pointer
        hiprtc program handle
    options : list of string
        option list to be passed to compilation
    """

    e_options = list()
    for option in options:
        e_options.append(option.encode('utf-8'))
    c_options = (ctypes.c_char_p * len(e_options))()
    c_options[:] = e_options
    status = _libhiprtc.hiprtcCompileProgram(prog, len(c_options), c_options)
    hiprtcCheckStatus(status)


_libhiprtc.hiprtcGetProgramLogSize.restype = int
_libhiprtc.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                               ctypes.POINTER(ctypes.c_size_t)]  # Size of log
_libhiprtc.hiprtcGetProgramLog.restype = int
_libhiprtc.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p,               # hiprtcProgram
                                           ctypes.POINTER(ctypes.c_char)]  # log


def hiprtcGetProgramLog(prog):
    """
    Gets the hiprtc Log
    Parameters
    ----------
    prog : ctypes pointer
        hiprtc program handle
    Returns
    -------
    log : string
        program compilation log (warnings/errors)
    """
    log_size = ctypes.c_size_t()
    status = _libhiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
    hiprtcCheckStatus(status)

    log = "0" * log_size.value
    e_log = log.encode('utf-8')
    status = _libhiprtc.hiprtcGetProgramLog(prog, e_log)
    hiprtcCheckStatus(status)
    return e_log


_libhiprtc.hiprtcGetCodeSize.restype = int
_libhiprtc.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p,                 # hiprtcProgram
                                         ctypes.POINTER(ctypes.c_size_t)]  # Size of log
_libhiprtc.hiprtcGetCode.restype = int
_libhiprtc.hiprtcGetCode.argtypes = [ctypes.c_void_p,               # hiprtcProgram
                                     ctypes.POINTER(ctypes.c_char)]  # log


def hiprtcGetCode(prog):
    """
    Gets the hiprtc compiled code
    Parameters
    ----------
    prog : ctypes pointer
        hiprtc program handle
    Returns
    -------
    code : string
        hiprtc module code
    """
    code_size = ctypes.c_size_t()
    status = _libhiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
    hiprtcCheckStatus(status)

    code = "0" * code_size.value
    e_code = code.encode('utf-8')
    status = _libhiprtc.hiprtcGetCode(prog, e_code)
    hiprtcCheckStatus(status)
    return e_code


_libhip.hipGetErrorString.restype = ctypes.c_char_p
_libhip.hipGetErrorString.argtypes = [ctypes.c_int]


def hiprtcGetErrorString(e):
    """
    Retrieve hip error string.
    Return the string associated with the specified hiprtc error status
    code.
    Parameters
    ----------
    e : int
        Error number.
    Returns
    -------
    s : str
        Error string.
    """

    return _libhip.hipGetErrorString(e)

# Generic hip error


class hipError(Exception):
    """hip error"""
    pass


class hipErrorInvalidValue(hipError):
    __doc__ = _libhip.hipGetErrorString(1)
    pass


class hipErrorOutOfMemory(hipError):
    __doc__ = _libhip.hipGetErrorString(2)
    pass


class hipErrorNotInitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(3)
    pass


class hipErrorDeinitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(4)
    pass


class hipErrorProfilerDisabled(hipError):
    __doc__ = _libhip.hipGetErrorString(5)
    pass


class hipErrorProfilerNotInitialized(hipError):
    __doc__ = _libhip.hipGetErrorString(6)
    pass


class hipErrorProfilerAlreadyStarted(hipError):
    __doc__ = _libhip.hipGetErrorString(7)
    pass


class hipErrorProfilerAlreadyStopped(hipError):
    __doc__ = _libhip.hipGetErrorString(8)
    pass


class hipErrorInvalidConfiguration(hipError):
    __doc__ = _libhip.hipGetErrorString(9)
    pass


class hipErrorInvalidSymbol(hipError):
    __doc__ = _libhip.hipGetErrorString(13)
    pass


class hipErrorInvalidDevicePointer(hipError):
    __doc__ = _libhip.hipGetErrorString(17)
    pass


class hipErrorInvalidMemcpyDirection(hipError):
    __doc__ = _libhip.hipGetErrorString(21)
    pass


class hipErrorInsufficientDriver(hipError):
    __doc__ = _libhip.hipGetErrorString(35)
    pass


class hipErrorMissingConfiguration(hipError):
    __doc__ = _libhip.hipGetErrorString(52)
    pass


class hipErrorPriorLaunchFailure(hipError):
    __doc__ = _libhip.hipGetErrorString(53)
    pass


class hipErrorInvalidDeviceFunction(hipError):
    __doc__ = _libhip.hipGetErrorString(98)
    pass


class hipErrorNoDevice(hipError):
    __doc__ = _libhip.hipGetErrorString(100)
    pass


class hipErrorInvalidDevice(hipError):
    __doc__ = _libhip.hipGetErrorString(101)
    pass


class hipErrorInvalidImage(hipError):
    __doc__ = _libhip.hipGetErrorString(200)
    pass


class hipErrorInvalidContext(hipError):
    __doc__ = _libhip.hipGetErrorString(201)
    pass


class hipErrorContextAlreadyCurrent(hipError):
    __doc__ = _libhip.hipGetErrorString(202)
    pass


class hipErrorMapFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(205)
    pass


class hipErrorUnmapFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(206)
    pass


class hipErrorArrayIsMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(207)
    pass


class hipErrorAlreadyMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(208)
    pass


class hipErrorNoBinaryForGpu(hipError):
    __doc__ = _libhip.hipGetErrorString(209)
    pass


class hipErrorAlreadyAcquired(hipError):
    __doc__ = _libhip.hipGetErrorString(210)
    pass


class hipErrorNotMapped(hipError):
    __doc__ = _libhip.hipGetErrorString(211)
    pass


class hipErrorNotMappedAsArray(hipError):
    __doc__ = _libhip.hipGetErrorString(212)
    pass


class hipErrorNotMappedAsPointer(hipError):
    __doc__ = _libhip.hipGetErrorString(213)
    pass


class hipErrorECCNotCorrectable(hipError):
    __doc__ = _libhip.hipGetErrorString(214)
    pass


class hipErrorUnsupportedLimit(hipError):
    __doc__ = _libhip.hipGetErrorString(215)
    pass


class hipErrorContextAlreadyInUse(hipError):
    __doc__ = _libhip.hipGetErrorString(216)
    pass


class hipErrorPeerAccessUnsupported(hipError):
    __doc__ = _libhip.hipGetErrorString(217)
    pass


class hipErrorInvalidKernelFile(hipError):
    __doc__ = _libhip.hipGetErrorString(218)
    pass


class hipErrorInvalidGraphicsContext(hipError):
    __doc__ = _libhip.hipGetErrorString(219)
    pass


class hipErrorInvalidSource(hipError):
    __doc__ = _libhip.hipGetErrorString(300)
    pass


class hipErrorFileNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(301)
    pass


class hipErrorSharedObjectSymbolNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(302)
    pass


class hipErrorSharedObjectInitFailed(hipError):
    __doc__ = _libhip.hipGetErrorString(303)
    pass


class hipErrorOperatingSystem(hipError):
    __doc__ = _libhip.hipGetErrorString(304)
    pass


class hipErrorInvalidHandle(hipError):
    __doc__ = _libhip.hipGetErrorString(400)
    pass


class hipErrorNotFound(hipError):
    __doc__ = _libhip.hipGetErrorString(500)
    pass


class hipErrorNotReady(hipError):
    __doc__ = _libhip.hipGetErrorString(600)
    pass


class hipErrorIllegalAddress(hipError):
    __doc__ = _libhip.hipGetErrorString(700)
    pass


class hipErrorLaunchOutOfResources(hipError):
    __doc__ = _libhip.hipGetErrorString(701)
    pass


class hipErrorLaunchTimeOut(hipError):
    __doc__ = _libhip.hipGetErrorString(702)
    pass


class hipErrorPeerAccessAlreadyEnabled(hipError):
    __doc__ = _libhip.hipGetErrorString(704)
    pass


class hipErrorPeerAccessNotEnabled(hipError):
    __doc__ = _libhip.hipGetErrorString(705)
    pass


class hipErrorSetOnActiveProcess(hipError):
    __doc__ = _libhip.hipGetErrorString(708)
    pass


class hipErrorAssert(hipError):
    __doc__ = _libhip.hipGetErrorString(710)
    pass


class hipErrorHostMemoryAlreadyRegistered(hipError):
    __doc__ = _libhip.hipGetErrorString(712)
    pass


class hipErrorHostMemoryNotRegistered(hipError):
    __doc__ = _libhip.hipGetErrorString(713)
    pass


class hipErrorLaunchFailure(hipError):
    __doc__ = _libhip.hipGetErrorString(719)
    pass


class hipErrorCooperativeLaunchTooLarge(hipError):
    __doc__ = _libhip.hipGetErrorString(720)
    pass


class hipErrorNotSupported(hipError):
    __doc__ = _libhip.hipGetErrorString(801)
    pass


class hipErrorUnknown(hipError):
    __doc__ = _libhip.hipGetErrorString(999)
    pass


class hipErrorRuntimeMemory(hipError):
    __doc__ = _libhip.hipGetErrorString(1052)
    pass


class hipErrorRuntimeOther(hipError):
    __doc__ = _libhip.hipGetErrorString(1053)
    pass


hipExceptions = {
    1: hipErrorInvalidValue,
    2: hipErrorOutOfMemory,
    3: hipErrorNotInitialized,
    4: hipErrorDeinitialized,
    5: hipErrorProfilerDisabled,
    6: hipErrorProfilerNotInitialized,
    7: hipErrorProfilerAlreadyStarted,
    8: hipErrorProfilerAlreadyStopped,
    9: hipErrorInvalidConfiguration,
    13: hipErrorInvalidSymbol,
    17: hipErrorInvalidDevicePointer,
    21: hipErrorInvalidMemcpyDirection,
    35: hipErrorInsufficientDriver,
    52: hipErrorMissingConfiguration,
    53: hipErrorPriorLaunchFailure,
    98: hipErrorInvalidDeviceFunction,
    100: hipErrorNoDevice,
    101: hipErrorInvalidDevice,
    200: hipErrorInvalidImage,
    201: hipErrorInvalidContext,
    202: hipErrorContextAlreadyCurrent,
    205: hipErrorMapFailed,
    206: hipErrorUnmapFailed,
    207: hipErrorArrayIsMapped,
    208: hipErrorAlreadyMapped,
    209: hipErrorNoBinaryForGpu,
    210: hipErrorAlreadyAcquired,
    211: hipErrorNotMapped,
    212: hipErrorNotMappedAsArray,
    213: hipErrorNotMappedAsPointer,
    214: hipErrorECCNotCorrectable,
    215: hipErrorUnsupportedLimit,
    216: hipErrorContextAlreadyInUse,
    217: hipErrorPeerAccessUnsupported,
    218: hipErrorInvalidKernelFile,
    219: hipErrorInvalidGraphicsContext,
    300: hipErrorInvalidSource,
    301: hipErrorFileNotFound,
    302: hipErrorSharedObjectSymbolNotFound,
    303: hipErrorSharedObjectInitFailed,
    304: hipErrorOperatingSystem,
    400: hipErrorInvalidHandle,
    500: hipErrorNotFound,
    600: hipErrorNotReady,
    700: hipErrorIllegalAddress,
    701: hipErrorLaunchOutOfResources,
    702: hipErrorLaunchTimeOut,
    704: hipErrorPeerAccessAlreadyEnabled,
    705: hipErrorPeerAccessNotEnabled,
    708: hipErrorSetOnActiveProcess,
    710: hipErrorAssert,
    712: hipErrorHostMemoryAlreadyRegistered,
    713: hipErrorHostMemoryNotRegistered,
    719: hipErrorLaunchFailure,
    720: hipErrorCooperativeLaunchTooLarge,
    801: hipErrorNotSupported,
    999: hipErrorUnknown,
    1052: hipErrorRuntimeMemory,
    1053: hipErrorRuntimeOther
}


def hipCheckStatus(status):
    """
    Raise hip exception.
    Raise an exception corresponding to the specified hip runtime error
    code.
    Parameters
    ----------
    status : int
        hip runtime error code.
    See Also
    --------
    hipExceptions
    """

    if status != 0:
        try:
            e = hipExceptions[status]
        except KeyError:
            raise hipError('unknown hip error %s' % status)
        else:
            raise e


# Memory allocation functions (adapted from pystream):
_libhip.hipMalloc.restype = int
_libhip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                              ctypes.c_size_t]


def hipMalloc(count, ctype=None):
    """
    Allocate device memory.
    Allocate memory on the device associated with the current active
    context.
    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.
    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    ptr = ctypes.c_void_p()
    status = _libhip.hipMalloc(ctypes.byref(ptr), count)
    hipCheckStatus(status)
    if ctype != None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr


_libhip.hipFree.restype = int
_libhip.hipFree.argtypes = [ctypes.c_void_p]


def hipFree(ptr):
    """
    Free device memory.
    Free allocated memory on the device associated with the current active
    context.
    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    status = _libhip.hipFree(ptr)
    hipCheckStatus(status)


_libhip.hipMallocPitch.restype = int
_libhip.hipMallocPitch.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_size_t),
                                   ctypes.c_size_t, ctypes.c_size_t]


def hipMallocPitch(pitch, rows, cols, elesize):
    """
    Allocate pitched device memory.
    Allocate pitched memory on the device associated with the current active
    context.
    Parameters
    ----------
    pitch : int
        Pitch for allocation.
    rows : int
        Requested pitched allocation height.
    cols : int
        Requested pitched allocation width.
    elesize : int
        Size of memory element.
    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    """

    ptr = ctypes.c_void_p()
    status = _libhip.hipMallocPitch(ctypes.byref(ptr),
                                    ctypes.c_size_t(pitch), cols*elesize,
                                    rows)
    hipCheckStatus(status)
    return ptr, pitch


# Memory copy modes:
hipMemcpyHostToHost = 0
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
hipMemcpyDeviceToDevice = 3
hipMemcpyDefault = 4

_libhip.hipMemcpy.restype = int
_libhip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int]


def hipMemcpy_htod(dst, src, count):
    """
    Copy memory from host to device.
    Copy data from host memory to device memory.
    Parameters
    ----------
    dst : ctypes pointer
        Device memory pointer.
    src : ctypes pointer
        Host memory pointer.
    count : int
        Number of bytes to copy.
    """

    status = _libhip.hipMemcpy(dst, src,
                               ctypes.c_size_t(count),
                               hipMemcpyHostToDevice)
    hipCheckStatus(status)


def hipMemcpy_dtoh(dst, src, count):
    """
    Copy memory from device to host.
    Copy data from device memory to host memory.
    Parameters
    ----------
    dst : ctypes pointer
        Host memory pointer.
    src : ctypes pointer
        Device memory pointer.
    count : int
        Number of bytes to copy.
    """

    status = _libhip.hipMemcpy(dst, src,
                               ctypes.c_size_t(count),
                               hipMemcpyDeviceToHost)
    hipCheckStatus(status)


_libhip.hipMemGetInfo.restype = int
_libhip.hipMemGetInfo.argtypes = [ctypes.c_void_p,
                                  ctypes.c_void_p]


def hipMemGetInfo():
    """
    Return the amount of free and total device memory.
    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.
    """

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    status = _libhip.hipMemGetInfo(ctypes.byref(free),
                                   ctypes.byref(total))
    hipCheckStatus(status)
    return free.value, total.value


_libhip.hipSetDevice.restype = int
_libhip.hipSetDevice.argtypes = [ctypes.c_int]


def hipSetDevice(dev):
    """
    Set current hip device.
    Select a device to use for subsequent hip operations.
    Parameters
    ----------
    dev : int
        Device number.
    """

    status = _libhip.hipSetDevice(dev)
    hipCheckStatus(status)


_libhip.hipGetDevice.restype = int
_libhip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]


def hipGetDevice():
    """
    Get current hip device.
    Return the identifying number of the device currently used to
    process hip operations.
    Returns
    -------
    dev : int
        Device number.
    """

    dev = ctypes.c_int()
    status = _libhip.hipGetDevice(ctypes.byref(dev))
    hipCheckStatus(status)
    return dev.value


# Memory types:
hipMemoryTypeHost = 1
hipMemoryTypeDevice = 2


class hipPointerAttributes(ctypes.Structure):
    _fields_ = [
        ('memoryType', ctypes.c_int),
        ('device', ctypes.c_int),
        ('devicePointer', ctypes.c_void_p),
        ('hostPointer', ctypes.c_void_p)
    ]


_libhip.hipPointerGetAttributes.restype = int
_libhip.hipPointerGetAttributes.argtypes = [ctypes.c_void_p,
                                            ctypes.c_void_p]


def hipPointerGetAttributes(ptr):
    """
    Get memory pointer attributes.
    Returns attributes of the specified pointer.
    Parameters
    ----------
    ptr : ctypes pointer
        Memory pointer to examine.
    Returns
    -------
    memory_type : int
        Memory type; 1 indicates host memory, 2 indicates device
        memory.
    device : int
        Number of device associated with pointer.
    """

    attributes = hipPointerAttributes()
    status = \
        _libhip.hipPointerGetAttributes(ctypes.byref(attributes), ptr)
    hipCheckStatus(status)
    return attributes.memoryType, attributes.device


_libhip.hipModuleLoadData.restype = int
_libhip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # Module
                                      ctypes.c_void_p]                 # Image


def hipModuleLoadData(data):
    """
    Load hip module data
    Parameters
    ----------
    data : ctypes pointer
        Memory pointer to load.
    Returns
    -------
    module : ctypes ptr
        hip module
    """

    module = ctypes.c_void_p()
    status = _libhip.hipModuleLoadData(ctypes.byref(module), data)
    hipCheckStatus(status)
    return module


_libhip.hipModuleGetFunction.restype = int
_libhip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),  # Kernel
                                         ctypes.c_void_p,                    # Module
                                         ctypes.POINTER(ctypes.c_char)]      # kenrel name


def hipModuleGetFunction(module, func_name):
    """
    gets the kernel from module
    Parameters
    ----------
    module : ctypes ptr
        pointer to created module
    func_name : string
        name of function to be retrived from module
    Returns
    -------
    kernel : ctypes ptr
        pointer to kernel type
    """
    e_func_name = func_name.encode('utf-8')
    kernel = ctypes.c_void_p()
    status = _libhip.hipModuleGetFunction(
        ctypes.byref(kernel), module, e_func_name)
    hipCheckStatus(status)
    return kernel


_libhip.hipModuleUnload.restype = int
_libhip.hipModuleUnload.argtypes = [ctypes.c_void_p]


def hipModuleUnload(module):
    """
    gets the kernel from module
    Parameters
    ----------
    module : ctypes ptr
        pointer to created module
    """
    status = _libhip.hipModuleUnload(module)
    hipCheckStatus(status)


_libhip.hipModuleLaunchKernel.restype = int
_libhip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p,                 # kernel
                                          ctypes.c_uint,                   # block x
                                          ctypes.c_uint,                   # block y
                                          ctypes.c_uint,                   # block z
                                          ctypes.c_uint,                   # thread x
                                          ctypes.c_uint,                   # thread y
                                          ctypes.c_uint,                   # thread z
                                          ctypes.c_uint,                   # shared mem
                                          ctypes.c_void_p,                 # stream
                                          # kernel params
                                          ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_void_p)]  # extra


def hipModuleLaunchKernel(kernel, bx, by, bz, tx, ty, tz, shared, stream, struct):
    """
    Launch the kernel
    Parameters
    ----------
    kernel : ctypes ptr
        kernel from loaded module
    bx : int
        dim x
    by : int
        dim y
    bz : int
        dim z
    tx : int
        dim x
    ty : int
        dim y
    tz : int
        dim z
    shared : int
        shared mem
    stream : ctype void ptr
        stream object
    struct : ctypes structure
        struct of packed up arguments of kernel
    """
    c_bx = ctypes.c_uint(bx)
    c_by = ctypes.c_uint(by)
    c_bz = ctypes.c_uint(bz)
    c_tx = ctypes.c_uint(tx)
    c_ty = ctypes.c_uint(ty)
    c_tz = ctypes.c_uint(tz)
    c_shared = ctypes.c_uint(shared)

    # ctypes.sizeof(struct)
    # hip_launch_param_buffer_ptr = ctypes.c_void_p(1)
    # hip_launch_param_buffer_size = ctypes.c_void_p(2)
    # hip_launch_param_buffer_end = ctypes.c_void_p(3)
    # size = ctypes.c_size_t(ctypes.sizeof(struct))
    # p_size = ctypes.c_void_p(ctypes.addressof(size))
    # p_struct = ctypes.c_void_p(ctypes.addressof(struct))
    # config = (ctypes.c_void_p * 5)(hip_launch_param_buffer_ptr, p_struct,
    #                                hip_launch_param_buffer_size, p_size, hip_launch_param_buffer_end)
    # nullptr = ctypes.POINTER(ctypes.c_void_p)(ctypes.c_void_p(0))

    status = _libhip.hipModuleLaunchKernel(
        kernel, c_bx, c_by, c_bz, c_tx, c_ty, c_tz, c_shared, stream, None, struct)
    hipCheckStatus(status)


_libhip.hipDeviceSynchronize.restype = int
_libhip.hipDeviceSynchronize.argtypes = []


def hipDeviceSynchronize():
    """
    Device level sync
    """
    status = _libhip.hipDeviceSynchronize()
    hipCheckStatus(status)


_libhip.hipInit.restype = int
_libhip.hipInit.restype = [ctypes.c_uint]  # flags


def hipInit(flags):
    """
    Explicitly initializes the HIP runtime.

    Most HIP APIs implicitly initialize the HIP runtime.
    This API provides control over the timing of the initialization.

    Parameters
    ----------
    flags : int
    """
    c_flags = ctypes.c_uint(flags)
    status = _libhip.hipInit(c_flags)
    hipCheckStatus(status)


def _arg_size(arg_info: ArgInfo):
    return arg_info.element_num_bytes * reduce(lambda x, y: x*y, arg_info.numpy_shape)


def initialize_rocm():
    # Initialize ROCM Driver API
    hipInit(0)


def compile_rocm_program(rocm_src_path: pathlib.Path, func_name):
    src = rocm_src_path.read_text()

    prog = hiprtcCreateProgram(source=src, name=func_name, header_names=HEADER_MAP.keys(
    ), header_sources=HEADER_MAP.values())
    hiprtcCompileProgram(prog, ['--offload-arch=gfx906'])
    code = hiprtcGetCode(prog)

    return code


def get_func_from_rocm_program(rocm_program, func_name):
    rocm_module = hipModuleLoadData(rocm_program)
    kernel = hipModuleGetFunction(rocm_module, func_name)
    return kernel


def allocate_rocm_mem(arg_infos: List[ArgInfo]):
    device_mem = []
    for arg in arg_infos:
        mem = hipMalloc(_arg_size(arg))
        device_mem.append(mem)

    return device_mem


def transfer_mem_host_to_cuda(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'input' in arg_info.usage:
            hipMemcpy_htod(dst=device_arg, src=host_arg.ctypes.data,
                           count=_arg_size(arg_info))


def transfer_mem_cuda_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'output' in arg_info.usage:
            hipMemcpy_dtoh(dst=host_arg.ctypes.data, src=device_arg,
                           count=_arg_size(arg_info))


def device_args_to_ptr_list(device_args: List):
    ptrs = [
        np.array([int(d_arg)], dtype=np.uint64) for d_arg in device_args
    ]
    ptrs = np.array([ptr.ctypes.data for ptr in ptrs], dtype=np.uint64)

    return ptrs


def create_loader_for_device_function(device_func, hat_details):
    hat_path: pathlib.Path = hat_details.path
    rocm_src_path: pathlib.Path = hat_path.parent / device_func["provider"]
    func_name = device_func["name"]

    rocm_program = compile_rocm_program(rocm_src_path, func_name)

    initialize_rocm()

    kernel = get_func_from_rocm_program(rocm_program, func_name)

    hat_arg_descriptions = device_func["arguments"]
    arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
    launch_parameters = device_func["launch_parameters"]

    def f(*args):
        verify_args(args, arg_infos, func_name)
        device_mem = allocate_rocm_mem(arg_infos)
        transfer_mem_host_to_rocm(
            device_args=device_mem, host_args=args, arg_infos=arg_infos)
        ptrs = device_args_to_ptr_list(device_mem)

        # err, stream = cuda.cuStreamCreate(0)
        # ASSERT_DRV(err)

        hipModuleLaunchKernel(
            kernel,
            *launch_parameters,  # [ grid[x-z], block[x-z] ]
            0,    # dynamic shared memory
            0,    # stream
            ptrs.ctypes.data,    # data
        )
        hipDeviceSynchronize()
        # cuStreamSynchronize()
        # ASSERT_DRV(err)

        transfer_mem_rocm_to_host(
            device_args=device_mem, host_args=args, arg_infos=arg_infos)

    return f
