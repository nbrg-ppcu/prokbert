

FORWARD = "forward"
BACKWARD = "backward"
RANDOM = "random"
CONTIGUOUS = "contiguous"

RC_TABLE = str.maketrans({
    "A":"T","C":"G","G":"C","T":"A","U":"A",
    "R":"Y","Y":"R","S":"S","W":"W","K":"M","M":"K",
    "B":"V","D":"H","H":"D","V":"B","N":"N",
    "a":"t","c":"g","g":"c","t":"a","u":"a",
    "r":"y","y":"r","s":"s","w":"w","k":"m","m":"k",
    "b":"v","d":"h","h":"d","v":"b","n":"n",
})