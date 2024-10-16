from django import template

register = template.Library()


# register your custom template tag
@register.simple_tag
def item_counter(outerloop_counter0, innerloop_counter0):
    return (outerloop_counter0 * 9) + innerloop_counter0


@register.filter
def index(items, i):
    return items[int(i)]

@register.filter(name='to')
def to(value, arg):
    return range(value, arg + 1)

@register.filter
def multiply(value, arg):
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return ''
    
@register.filter
def add(value, arg):
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        return ''

@register.filter
def subtract(value, arg):
    try:
        return int(value) - int(arg)
    except (ValueError, TypeError):
        return ''