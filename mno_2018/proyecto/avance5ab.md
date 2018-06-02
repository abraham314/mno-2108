**Abraham:** 
`instalación de cuda versión 9.1 para compilar el programa svd.cu, existe un error a la hora de ejecutar dicho programa para obtener las matrices U,V.T y d_s para la descomposición, el error que muestra es : GPUassert: unknown error Utilities.cu 37, el detalle de ejcución se puede ver en esta imagen.` 
[Ejecucion] (https://drive.google.com/open?id=1-QiBuk11H_ZwF_7gMZWzC30Q8CI_ineU).
`El error viene de de uno de los archivos de dependencias que es Utilities.cu donde se define la función gpuAssert  la cual se explica en esta liga` [gpuAssert] (https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api) `que es una función que verifica si hay errores en el código de la API runtime, si existe un error se envía un mensaje de texto que describe el error y la línea donde ocurrió, ene este caso parece ser en archivo Utilities.cu ene esta linea "extern "C" void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }", todo se esta ejecutando a nivel local.`  

**Equipo:** 
`Avanzamos en la documentación del proyecto y estamos probando la ejecución del código svd.cu aunque con algunos problemas, el objetivo de usar ese código(original y/o modificado) es que se haga la descomposión sobre la matriz de movilens, estamos investigando sobre el error del programa svd.cu.`  

