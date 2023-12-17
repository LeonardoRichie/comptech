import basic

def process_input(text):
    if text.strip() == "":
        return None, None  # Skipping empty input
    
    result, error = basic.run('<stdin>', text)

    if error:
        return None, error.as_Str_()
    elif result:
        if len(result.elements) == 1:
            return repr(result.elements[0]), None
        else:
            return repr(result), None

while True:
    user_input = input('language > ')
    output, error_message = process_input(user_input)

    if error_message:
        print(error_message)
    elif output:
        print(output)

    #menit 9:25var = 5
    