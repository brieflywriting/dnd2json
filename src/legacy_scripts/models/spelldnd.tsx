import { Spell, ID, Reference } from "./interfaces";
enum SpellSchool {
    Evocation = 1,
    Conjuration,
    Abjuration, Enchantment, Divination, Illusion, Necromancy, Transmutation, unknown
}
export interface SpellDnd extends Spell {
    id: ID;
    name: string;
    circle_level: number;
    school: SpellSchool;
    casting_time: string;
    range: number;
    components: string[];
    duration: string;
    description: string;
    higher_levels?: string;
    classes?: Reference[];
    credits?: string;
}