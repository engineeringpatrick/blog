import { classNames } from "../util/lang";
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types";

const Image: QuartzComponent = ({displayClass}: QuartzComponentProps) => {
  return (
      <a 
        href={`https://www.weforum.org/stories/2021/12/humans-multiplanetary-species`}
        target={`_blank`}
        >
        <img 
            class={classNames(displayClass, "image")} 
            src={`/static/astronaut.jpeg`}
        />
    </a>
  )
}
export default (() => Image) satisfies QuartzComponentConstructor;
