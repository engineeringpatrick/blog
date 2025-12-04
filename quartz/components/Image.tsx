import { classNames } from "../util/lang";
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types";

const Image: QuartzComponent = ({displayClass}: QuartzComponentProps) => {
  return (
      <a 
        class={classNames(displayClass, "sidebar-a")}
        href={`https://www.weforum.org/stories/2021/12/humans-multiplanetary-species`}
        target={`_blank`}
        >
        <img 
            class={classNames(displayClass, "sidebar-img")} 
            src={`/static/astronaut.jpeg`}
        />
    </a>
  )
}

Image.css = `
.sidebar-a {
  width: 100%;
  height: 100vh;
  overflow: hidden;
}
.sidebar-img {
  opacity: 0.5;
  object-fit: cover;
}
`
export default (() => Image) satisfies QuartzComponentConstructor;
