def number_to_words(num: int) -> str:
    """Convert an integer into it's English words representation (short scale),
    and return the result.\n
    Support any integer from range ```-10**36 < x < 10**36.``` \n
    The first letter of every word included in the result is capitalized.

    Example:
        >>> number_to_words(1234)
        "One Thousand Two Hundred Thirty Four"
        number_to_words(-974258)
        "Negative Nine Hundred Seventy Four Thousand Two Hundred Fifty Eight"

    """
    if num == 0:
        return "Zero" 
    word = []
    if num < 0:
        word.append("Negative")
        num = abs(num)

    # Indexes start at 0, not 1, therefore an empty string is added at the start 
    # so the number matches the index.
    below_twenty = [
        "", "One", "Two", "Three", "Four", "Five",
        "Six", "Seven", "Eight", "Nine", "Ten",
        "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
        "Sixteen", "Seventeen", "Eighteen", "Nineteen" 
    ]

    # Same as above. There is no "Zeroty" or "Tenty" in English.
    tens = [
        "", "", "Twenty", "Thirty", "Forty",
        "Fifty", "Sixty", "Seventy",
        "Eighty", "Ninety"
    ]

    scales = [
        "Thousand", "Million", "Billion", 
        "Trillion", "Quadrillion", "Quintillion", 
        "Sextillion", "Septillion", "Octillion",
        "Nonillion", "Decillion"
        ][::-1]
    scale = 1000**len(scales)

    # Use recursion to convert an integer x into it's English representation, where 0 < x < 1000.
    def helper(number): 
        if number >= 100:
            return " ".join([below_twenty[number//100], "Hundred", helper(number % 100)]).strip()
        elif number >= 20:
            return " ".join([tens[number//10], helper(number % 10)]).strip()
        else:
            return below_twenty[number]
    
    for x in scales:
        if num >= scale:

            # Determine the amount of the same scale used in that number,
            # Then adds the correct English representation word for that scale.
            # However, this loop does not include the last three digits.

            word.extend([helper(num // scale), x])
            num %= scale
        scale //= 1000 

    # This guarantees the last three digits are included, as `num` is now less than 1000.
    word.append(helper(num))  
    return " ".join(word).strip()

n = int(input())
print(number_to_words(n))
print(n)