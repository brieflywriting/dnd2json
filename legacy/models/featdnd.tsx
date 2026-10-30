import { ID, Reference, Feat } from "./interfaces";

export interface FeatDnD extends Feat {
    id: ID;
    name: string;
    description: string;
    level_acquired: number;
    prerequisites?: Reference[]; // e.g. [{ id: "class:warlock" }]
    source: ID;
    credits?: string;

}