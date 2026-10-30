import { Class, Reference } from "./interfaces";


export interface ClassStatistics {
    hit_die: string;
    proficiencies: {
        armor: string[];
        weapons: string[];
        tools: string[];
        saving_throws: string[];
        skills: string[];
    };
    ability_dc?: {
        ability: number;
        formula?: string;
    };
}

export interface ClassEquipment {
    starting_equipment: Reference[];
    wealth_roll?: string;
}

export interface ClassDnD extends Class {
    id: Class;
    name: string;
    description: string;
    statistics: ClassStatistics;
    equipment: ClassEquipment;
    progressionTable?: any;
    feats: Reference[];               // e.g. [{ id: "feat:eldritch-invocations" }]
    subclasses: Reference[];
    spells: Reference[];
    homebrewSpells?: Reference[];
    credits?: string;
}