from typing import Callable, Tuple
import graphene
from grapple.helpers import register_streamfield_block
from grapple.models import GraphQLStreamfield, GraphQLString, GraphQLField
from grapple.registry import registry as grapple_registry
from wagtail import blocks


class CardListCardBlock(blocks.StructBlock):
    title = blocks.CharBlock(required=True)
    short_description = blocks.CharBlock(required=False)

    graphql_fields = [
        GraphQLString('title'),
        GraphQLString('short_description'),
    ]



def build_block_type(cls: type, type_prefix: str, interfaces: Tuple[graphene.Interface, ...], base_type=graphene.ObjectType):
    from grapple.actions import build_streamfield_type
    ret = build_streamfield_type(cls, type_prefix, interfaces, base_type)  # type: ignore
    del ret._meta.fields['id']  # type: ignore
    return ret


class GraphQLBlockField(GraphQLField):
    def __init__(
        self,
        field_name: str,
        block_cls: type[blocks.Block],
        *,
        required: bool | None = None,
        **kwargs
    ):
        def build_type():
            return build_block_type(block_cls, '', interfaces=())
        super().__init__(field_name, build_type, required=required, **kwargs)  # type: ignore


@register_streamfield_block
class CardListBlock(blocks.StructBlock):
    title = blocks.CharBlock(required=False)
    cards = blocks.ListBlock(CardListCardBlock(), required=True)

    graphql_fields = [
        GraphQLString('title'),
        GraphQLBlockField('cards', CardListCardBlock, is_list=True, required=True),
    ]
